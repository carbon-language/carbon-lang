#ifndef FLANG_SEMA_SCOPE_H
#define FLANG_SEMA_SCOPE_H

#include "flang/Sema/Identifier.h"
#include "flang/Sema/Type.h"
#include "flang/Sema/Attr.h"

#include <vector>

namespace Fortran {
namespace semantics {

class Scope ;

// 

// Describe a symbol.
//
//
//
class Symbol {
public:
  enum ClassId { 
    TemporarySymbolId,
    ParameterSymbolId, 
    SubprogramSymbolId,
    SubSymbolId,
    InterfaceSymbolId,
    ImportedSymbolId,
    UsedSymbolId,
    MemberSymbolId,
    DummyArgumentSymbolId,
    ExternalSymbolId,
    NamelistSymbolId,
    CommonSymbolId,
    VariableSymbolId,  
    DerivedTypeSymbolId,     
    // ... and probably more to come
    last_ClassId
  } ;
private:
  ClassId cid_; 
  Scope * owner_ ;          // The scope that owns this symbol
  const Identifier *name_;  // The local name of that symbol
 private:
 TypeSpec * type_ = nullptr;   // The type associated to that symbol 
public:
  Symbol(ClassId cid, Scope * owner, Identifier *name) :
    cid_(cid), owner_(owner), name_(name) {} 
  
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
  TemporarySymbol(Scope * owner, Identifier *name) : Symbol(TemporarySymbolId,owner,name) {} 
private:
  // If not NULL, then this is the actual symbol to be used instead of this symbol.
  Symbol *actual_ ; 
private:
};

// A symbol representing a parameter whose value can be queried.
//
//
class ParameterSymbol : public Symbol {
public:
  ParameterSymbol(Scope * owner, Identifier *name) : Symbol(ParameterSymbolId,owner,name) {} 
};

// A symbol representing an EXTERNAL function. 
class ExternalSymbol : public Symbol {
public:
  ExternalSymbol(Scope * owner, Identifier *name) : Symbol(ExternalSymbolId,owner,name) {} 
};

//
// A symbol representing a variable. 
//
// The variable may be local or part of a common.
// 
// Question: Do we want to represent pointers using VariableSymbol or a dedicated class? 
// 
class VariableSymbol : public Symbol {
public:
  VariableSymbol(Scope * owner, Identifier *name) : Symbol(VariableSymbolId,owner,name) {} 
};

// A symbol representing a dummy argument.
class DummyArgumentSymbol : public Symbol {
public:
  DummyArgumentSymbol(Scope * owner, Identifier *name) : Symbol(DummyArgumentSymbolId,owner,name) {} 
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
  CommonSymbol(Scope * owner, Identifier *name) : Symbol(CommonSymbolId,owner,name) {} 
private:
  // The content of the common section. 
  std::vector<Identifier *> content_ ;  
};

//
// A symbol describing a subprogram declared by a FUNCTION or SUBROUTINE construct
// (either as a main unit, a internal subprogram, or an interface).
//
// The SubprogramSymbol only occurs in the local scope providing the FUNCTION 
// or SUBROUTINE construct.
//
class SubprogramSymbol : public Symbol {
public:
private:
  Scope * inner_scope_ ;
  VariableSymbol *result ;  // Shortcut to the variable that holds the result 
  std::vector<DummyArgumentSymbol *> args ;
};

// Symbol describing an interface.
// A null name is allowed to represent an unnamed interface.
class InterfaceSymbol : public Symbol {
public:
  InterfaceSymbol(Scope * owner, Identifier *name) : Symbol(InterfaceSymbolId,owner,name) {} 
private:
  Scope * inner_scope_ ;
};

// A symbol imported from the parent or host scope of the local scope either
// automatically or via an 'import' statement.
class ImportedSymbol : public Symbol {
public:
  ImportedSymbol(Scope *owner, Identifier *name) : Symbol(ImportedSymbolId,owner,name) {} 
private:
  // Provide the target symbol in the parent or host scope.
  Symbol *target_ ; 
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
  SubSymbol(Scope * owner, Identifier *name) : Symbol(SubSymbolId,owner,name) {} 
private:
  // Provide the target symbol in the parent or host scope.
  Symbol *target_ ; 
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
  UsedSymbol(Scope * owner, Identifier *name) : Symbol(UsedSymbolId,owner,name) {} 
private:
  // Provide the list of target symbols in the 
  std::vector<Symbol*> targets_ ; 
  bool ambiguous_;
};


//
// A symbol representing a derived type.
//
class DerivedTypeSymbol : public Symbol{
public:
  DerivedTypeSymbol(Scope *owner, Identifier *name) : Symbol(DerivedTypeSymbolId,owner,name) {} 
private:
  Scope * inner_scope_ ;
};

  //
// A symbol representing a member in a Derived type
//
// Those symbols require a dedicated type because they are in a dedicated namespace.
// 
class MemberSymbol : public Symbol {
public:
  MemberSymbol(Scope * owner, Identifier *name) : Symbol(MemberSymbolId,owner,name) {} 
private:
};

//
// A symbol representing a namelist-group 

//
class NamelistSymbol : public Symbol {
public:
  NamelistSymbol(Scope * owner, Identifier *name) : Symbol(NamelistSymbolId,owner,name) {} 
private:
  std::vector<Symbol*> content_ ;
};

// NOTE:  Support for EQUIVALENCE is not 
//       
//

// =======================================================

//
// Describe a scope (or a symbol table)
//
// For now, there is only one class. Not sure if we need more. 
// Of course, there are multiple kinds of scopes in Fortran and 
// they all behave slightly differently. However, a single Scope class
// that adjust its behavior according to its ScopeKind is probably 
// easier to implement than a dozen of classes. 
//
// Remark: ImportedSymbol and UsedSymbol may be created on the fly the first time they 
// are referenced (in order to avoid an explosion of the number of symbols) 
//
class Scope 
{
public:

  enum ScopeKind { 
    SK_PROGRAM,      // a scope associated to a PROGRAM 
    SK_MODULE,       // a scope associated to a MODULE
    SK_SUBMODULE,    // a scope associated to a SUBMODULE
    SK_FUNCTION,     // a scope associated to a FUNCTION 
    SK_SUBROUTINE,   // a scope associated to a SUBROUTNE    
    SK_BLOCKDATA,    // a scope associated to a BLOCKDATA    
    SK_USE_MODULE,   // a scope describing all the public symbols of a module.   
    SK_BLOCK,        // a scope associated to a BLOCK construct
    SK_DERIVED,      // a scope associated to a derived TYPE construct
    SK_INTERFACE     // a scope associated to an interface

    // ... and probably more to come
  }  ;
      
private:

  ScopeKind kind_ ;

  // For scopes associated to a name (so most of them except BLOCK
  // and unnamed INTERFACE) provide the associated symbol in the 
  // parent scope.  
  Symbol * self_ ; 

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
    Scope * module ;
    // All identifiers to exclude when doing a lookup 
    // in that module
    std::vector<Identifier *> exclude ;
  };

  // 
  // The parent scope is the scope that lexically owns this local scope. 
  // Symbols are normally not searched is the parent scope except in the case of 
  // symbols declared by an 'import, only'. 
  //  

  Scope * parent_scope_ ;   
  
  // The host scope is the scope that is searched last. 9
  // 
  //
  //
  Scope * host_scope_ ;           // The host scope 

  std::list<Scope *> use_ ;       // Scopes for the modules that are entirely used   
  std::list<Scope *> use_only_ ;  // Scopes for the modules that are partially used  
 
  // All visible entries in the scope. 
  //
  // The entries are sorted by lexical order of declaration (except for
  // subprogram that are added at the end of the declaration part) which 
  // means that lookups must be performed in reverse order. 
  //
  // The reason to do that instead of using a map or any other data structure
  // optimized for fast lookup is to allow inner scopes (e.g. BLOCK) to see only a 
  // subset of their parent scope.
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

  std::vector<Symbol*> entries_ ; 

  // 
  // A vector containing all Symbols owned by that scope
  // 
  std::vector<Symbol*> owned_ ; 

  public:
  
  // The system scope is the scope that provides intrinsic subprograms
  
  Scope * getSystemScope() ; 

  Symbol *lookup(const Identifier *name) ; 

};

} // of namespace semantics
} // of namespace Fortran

#endif
