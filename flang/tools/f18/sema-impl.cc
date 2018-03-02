
#include "../../lib/parser/format-specification.h"
#include "../../lib/parser/idioms.h"
#include "../../lib/parser/indirection.h"
#include "../../lib/parser/parse-tree-visitor.h"
#include "../../lib/parser/parse-tree.h"

#include "flang/Sema/Scope.h"

#include <vector>
#include <map>
#include <stack>

namespace psr = Fortran::parser ;
namespace sema  = Fortran::semantics ;

#include "GetValue.h"

#define TODO  do { std::cerr << "NOT YET IMPLEMENTED " << __FILE__ << ":" << __LINE__ << "\n" ; exit(1) ; } while(0)
#define CONSUME(x) do { (void)x ; } while(0) 
#define TRACE_CALL()  do { std::cerr << "*** call " << __PRETTY_FUNCTION__ << "\n" ; } while(0)
#define TRACE(msg) do { std::cerr << msg << "\n" ; } while(0)
#define FAIL(msg)  do { std::cerr << "FATAL " << __FILE__ << ":" << __LINE__ << ":\n   " << msg << "\n" ; exit(1) ; } while(0)


// Initialize the pointer used to attach semantic information to each parser-tree node
//
// Ideally, the function should be called once at the begining of the corresponding Pre() 
// member in Pass1. However, in case a forward reference to the Semantic<> data would be
// required, no error will occur when setting strict=false.
//
template <typename T>  Fortran::semantics::Semantic<T> & initSema(const T &node, bool strict=true) { 
  if (node.s) {
    if (strict) 
      FAIL( "Duplicate call of " << __PRETTY_FUNCTION__ ) ;
    else
      return *(node.s); 
  }
  auto s = new Fortran::semantics::Semantic<T>( const_cast<T*>(&node) ) ;
  const_cast<T&>(node).s = s; 
  return *s ; 
} 

// Retreive the semantic information attached to a parser-tree node
template <typename T> Fortran::semantics::Semantic<T> & getSema(const T &node) { 
  assert(node.s) ; 
  return *(node.s) ;
} 


#if 1
#include "flang/Sema/Scope.h"
#include <type_traits>
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <string>
#include <stddef.h>


namespace Fortran::semantics 
{

using namespace Fortran::parser  ;

template <typename ParserClass> 
Semantic<ParserClass> &
sema(ParserClass *node) 
{
  if ( node->s == NULL ) {
   node->s = new Semantic<ParserClass>(node) ; 
  }
  return *(node->s); 
} 

template <typename P, typename T>  P * vget( T & x ) {  
  return std::get_if<P>(x.u) ;
}


template <typename P, typename T>  P * vget_i( T & x ) {
  if ( auto v = std::get_if<Indirection<P>>(&x.u) ) {
    return &(**v) ;
  } else {
    return nullptr ; 
  }   
}


class LabelTable 
{
private:
  struct Entry {
    // TODO: what to put here
    Provenance loc; 
  }; 

  std::map<int,Entry> entries_ ;
  
public:
  
  void add( int label , Provenance loc ) 
  { 
    if (label<1 || label>99999) return ; // Hoops! 
    auto &entry = entries_[label] ;
    entry.loc = loc ; 
  }

  bool find(int label, Provenance &loc)
  {
    
    auto it = entries_.find(label);
    if( it != entries_.end()) {
      Entry & entry{it->second}; 
      loc = entry.loc; 
      return true; 
    }
    return false;
  }

  void dump() 
  {    
    TRACE( "==== Label Table ====");
    for ( int i=1 ; i<=99999 ;i++) {
      Provenance p;
      if ( find(i,p) ) {
        TRACE( "  #" << i << " at " << p.offset() ) ;
      }
    }
    TRACE( "=====================");
  }

}; // of class LabelTable


//
// 
//
//
class LabelTableStack {
private:
  std::stack<LabelTable*> stack ; 
public:
  LabelTable *PushLabelTable( LabelTable *table ) 
  {
    assert(table!=NULL);
    stack.push(table);
    return table; 
  }

  void PopLabelTable( LabelTable *table ) 
  {
    assert( !stack.empty() ) ;
    assert( stack.top() == table ) ;
    stack.pop(); 
  }

  LabelTable & GetLabelTable() {
    assert( !stack.empty() ) ;
    return *stack.top() ;
  }
  
  bool NoLabelTable() {
    return stack.empty() ; 
  }

}; // of class LabelTableStack


//////////////////////////////////////////////////////////////////
// 
// Declare here the members of the Semantic<> information that will 
// be attached to each parse-tree class. The default is an empty struct.
//
// Here are a few common fields 
//  
//     Scope *scope_provider=0 ;     // For each node providing a new scope
//     int stmt_label=0 ;            // For each node that consumes a label
//
//////////////////////////////////////////////////////////////////


#define DEFINE_SEMANTIC(Class) template <> struct Semantic<Class> { Semantic<Class>(Class *node) {} 
#define END_SEMANTIC }

DEFINE_SEMANTIC(ProgramUnit)
END_SEMANTIC;

DEFINE_SEMANTIC(MainProgram)
  Scope *scope_provider=0 ; 
  LabelTable *label_table=0 ; 
END_SEMANTIC;

DEFINE_SEMANTIC(FunctionSubprogram)
  Scope *scope_provider=0 ; 
  LabelTable *label_table=0 ; 
END_SEMANTIC;

DEFINE_SEMANTIC(FunctionStmt)
  int stmt_label=0 ;
END_SEMANTIC;

DEFINE_SEMANTIC(EndFunctionStmt)
  int stmt_label=0 ;
END_SEMANTIC;

DEFINE_SEMANTIC(TypeDeclarationStmt)
  int stmt_label=0 ;
END_SEMANTIC;

DEFINE_SEMANTIC(AssignmentStmt)
  int stmt_label=0 ;
END_SEMANTIC;

DEFINE_SEMANTIC(PrintStmt)
  int stmt_label=0 ;
END_SEMANTIC;

DEFINE_SEMANTIC(ProgramStmt)
  int stmt_label=0 ;
END_SEMANTIC;

DEFINE_SEMANTIC(EndProgramStmt)
  int stmt_label=0 ;
END_SEMANTIC;

DEFINE_SEMANTIC(ImplicitStmt)
  int stmt_label=0 ;
END_SEMANTIC;


} // of namespace Fortran::semantics

//////////////////////////////////////////////////////////////////

using sema::Scope ;
using sema::LabelTable ;
using sema::LabelTableStack ;


namespace Fortran::parser { 


class Pass1 : public LabelTableStack {
  
public:
  
  Pass1() : current_label(-1) 
  {
    system_scope  = new Scope(Scope::SK_SYSTEM, nullptr, nullptr ) ; 
    unit_scope    = new Scope(Scope::SK_GLOBAL, system_scope, nullptr) ;
    current_scope = nullptr ;
  }  

public:

  int current_label; // hold the value of a statement label until it get consumed (-1 means none) 
  Provenance current_label_loc; 

  Scope * system_scope ; 
  Scope * unit_scope ; 
  Scope * current_scope ; 


   Scope *EnterScope(Scope *s) { 
    assert(s) ; 
    assert(s->getParentScope() == current_scope ) ;
    current_scope = s ; 
    TRACE("Entering Scope " << s->toString() );    
    return s;
  } 

  void LeaveScope(Scope::Kind k) {
    assert( current_scope->getKind() == k ) ; 
    TRACE("Leaving Scope " << current_scope->toString() );
    current_scope = current_scope->getParentScope() ; 
  }  

public:


public:
  
  // Trace the location and label of any x with an accessible Statement<> in its type.
  template <typename T> void TraceStatementInfo(const T &x) { 
    auto & s = GetStatementValue(x) ;
    // TODO: compilation will fail is 's' is not of type Statement<...>.
    // Do we have a type trait to detect Statement<>?  
    //  if constexpr ( s is a Statement<> ) {
    if ( s.label ) {
      TRACE("stmt: loc=" << s.provenance.offset() << " label=" << s.label ) ; 
    } else {
      TRACE("stmt: loc=" << s.provenance.offset()  ) ; 
    }
    // } else {  TRACE("stmt: none") ; } 
  } 


protected:

  void 
  CheckStatementName( const sema::Identifier *expect , const sema::Identifier *found , std::string ctxt, bool required ) 
  {
    
    if ( expect ) { 
      if ( found && found != expect ) {
        FAIL("Unexpected " << ctxt << " name '" << found->name() << "' (expected '" << expect->name() << "') ");
      }
    } else if ( found ) {
      FAIL("Unexpected " << ctxt << " name '" << found->name() );
    }
    
  }

  //
  // Consume a label produced by a previous Statement.
  // That function should be called exactly once between each pair of Statement<>.
  //
  // For now, the sole purpose of the 'stmt' is to provide a relevant type 
  // for __PRETTY_FUNCTION__ but the association <label,stmt> will eventually be stored 
  // somewhere. 
  //
  // I still haven't figured out how to do that efficiently.
  //
  // There is obviously the problem of the type that could be solved using a huge 
  // std::variant but there is also the problem that node addresses are still subject 
  // to change where the tree requires a rewrite. 
  //
  // Another way could be to store the label in the Semantic field of stmt.
  // That is relatively easy to do but that does not really solve the 
  // problem of matching a label with its target. 
  //
  template<typename T> 
  int ConsumeLabel(const T &stmt) 
  {
    if ( current_label == -1 ) {
      FAIL("No label to consume in " << __PRETTY_FUNCTION__ );
    } else {
      int label = current_label ;
      current_label = -1 ;

      auto &sema = getSema(stmt); 
      sema.stmt_label = label;
      
      if ( label != 0 ) {
        LabelTable & table = GetLabelTable() ;
        Provenance old_loc ; 
        if ( table.find(label, old_loc) ) {
          FAIL("Duplicate label " << label 
               << "at @" << current_label_loc.offset() 
               << "and @" << old_loc.offset() ) ;          
        } else {
          table.add( label, current_label_loc) ;
        }
      }

      return label ;
    }
  }


  

public:

  template <typename T> bool Pre(const T &x) { 
    TRACE( "*** fallback " << __PRETTY_FUNCTION__  ) ; 
    return true ;
  }
  
  template <typename T> void Post(const T &) { 
    TRACE( "*** fallback " << __PRETTY_FUNCTION__  ) ; 
  }
  
  // fallback for std::variant


  template <typename... A> bool Pre(const std::variant<A...> &) { 
    //std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
    return true;
  }
  
  template <typename... A> void Post(const std::variant<A...> &) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
  }
  
  // fallback for std::tuple

  template <typename... A> bool Pre(const std::tuple<A...> &) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
    return true;
  }

  template <typename... A> void Post(const std::tuple<A...> &) { 
    //  std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
  }

  // fallback for std::string

  bool Pre(const std::string &x) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
    return true ;
  }

  void Post(const std::string &) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
  }

  // fallback for Indirection<>

  template <typename T> bool Pre(const psr::Indirection<T> &x) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
    return true ;
  }

  template <typename T> void Post(const psr::Indirection<T> &) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
  }



  //  ========== Statement<>  ===========

  template <typename T>
  bool Pre(const psr::Statement<T> &x) { 
    if ( current_label != -1 ) {
      TRACE("*** Label " << current_label << " (" << current_label_loc.offset() << ") was not consumed in " << __PRETTY_FUNCTION__ );
    }
    current_label = 0 ; 
    current_label_loc = x.provenance ;
    if ( x.label.has_value() ) {
      //
      // TODO: The parser stores the label in a std::uint64_t but does not report overflow
      //       which means that the following labels are currently accepted as valid:
      //         18446744073709551617 = 2^64+1 = 1
      //         18446744073709551618 = 2^64+2 = 2
      //         ...
      //
      if ( 1 <= x.label.value() && x.label.value() <= 99999 ) {
        current_label = x.label.value() ; 
      } else {
        FAIL( "##### Illegal label value " << x.label.value() << " at @" << x.provenance.offset() ) ;
      }
    } 
    return true ; 
  }

  template <typename T>
  void Post(const psr::Statement<T> &x) { 
    if ( current_label!=-1 )  {
      TRACE("*** Label " << current_label << " (" << current_label_loc.offset() << ") was not consumed in " << __PRETTY_FUNCTION__ );
      current_label=-1 ;
    }
  }
  
  //  ========== ProgramUnit  ===========

  bool Pre(const ProgramUnit &x) { 
    TRACE_CALL() ;
    current_scope = unit_scope; 
    return true ; 
  }

  void Post(const ProgramUnit &x) { 
    TRACE_CALL() ;
  }

  //  ========== MainProgram  ===========

  bool Pre(const MainProgram &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 


    sema::ProgramSymbol * symbol{0};
    const ProgramStmt    * program_stmt = GetOptValue( std::get<x.PROG>(x.t) ) ;
    const EndProgramStmt & end_stmt     = GetValue( std::get<x.END>(x.t) ) ;

    const sema::Identifier * program_ident{0};
 
    if ( program_stmt ) {
      const std::string & name = program_stmt->v ;
      TRACE("program name = " << name ) ; 
      program_ident = sema::Identifier::get(name) ;
      symbol = new sema::ProgramSymbol( current_scope, program_ident ) ;
      TraceStatementInfo( std::get<x.PROG>(x.t) ) ;
    }

    // TODO: Should we create a symbol when there is no PROGRAM statement? 
    
    // Install the scope 
    sema.scope_provider = EnterScope( new Scope(Scope::SK_PROGRAM, current_scope, symbol) )  ; 

    // Install the label table
    sema.label_table = PushLabelTable( new LabelTable ) ;
    
    // Check the name consistancy
    const std::string * end_name = GetOptValue(end_stmt.v) ;
    const sema::Identifier * end_ident = end_name ? sema::Identifier::get(*end_name) : nullptr ;

    CheckStatementName(program_ident,end_ident,"program",false) ;

    // if ( program_ident ) { 
    //   if ( end_ident && program_ident != end_ident ) {
    //     FAIL("Unexpected end program name '" << end_ident->name() << "' (expected '" << program_ident->name() << "') ");
    //   }
    // } else if ( program_ident ) {
    //   FAIL("Unexpected end program name '" << end_ident->name() << "'");
    // }
    
    return true ; 
  }

  void Post(const MainProgram &x) { 
    auto &sema = getSema(x); 
    GetLabelTable().dump() ; 
    PopLabelTable(sema.label_table)  ;     
    LeaveScope(Scope::SK_PROGRAM)  ;     
    TRACE_CALL() ;
  }

  //  ========== FunctionSubprogram  ===========

  bool Pre(const FunctionSubprogram &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 

    const FunctionStmt    & function_stmt = GetValue(std::get<x.FUNC>(x.t)) ; 
    const EndFunctionStmt & end_stmt      = GetValue(std::get<x.END>(x.t)) ; 

    const std::string &function_name = std::get<1>(function_stmt.t) ; 
    const sema::Identifier *function_ident = sema::Identifier::get(function_name) ;

    // TODO: lookup for name conflict 
    sema::Symbol *lookup ;
    if ( current_scope->getKind() == Scope::SK_GLOBAL ) {
      lookup = current_scope->LookupProgramUnit(function_ident) ;
      if (lookup) FAIL("A unit '" << function_ident->name() << "' is already declared") ;
    } else {
      lookup = current_scope->LookupLocal(function_ident) ;
      // TODO: There are a few cases, a function redeclaration is not necessarily a problem.
      //       A typical example is a PRIVATE or PUBLIC statement in a module
      if (lookup) FAIL("A unit '" << function_ident->name() << "' is already declared") ;
    }
   
    auto symbol = new sema::FunctionSymbol( current_scope, function_ident ) ;
    sema.scope_provider = EnterScope( new Scope(Scope::SK_FUNCTION, current_scope, symbol) )  ; 

    // Install the label table
    sema.label_table = PushLabelTable( new LabelTable ) ;

    TraceStatementInfo( std::get<x.FUNC>(x.t) ) ;

    // Check the end function name 
    const std::string * end_name = GetOptValue(end_stmt.v) ;
    const sema::Identifier * end_ident = end_name ? sema::Identifier::get(*end_name) : nullptr ;

    CheckStatementName(function_ident,end_ident,"function",false) ;

    return true ; 
  }

  void Post(const FunctionSubprogram &x) { 
    TRACE_CALL() ;
    auto &sema = getSema(x); 
    GetLabelTable().dump() ; 
    PopLabelTable(sema.label_table)  ;     
    LeaveScope(Scope::SK_FUNCTION)  ;     
  }

  //  ========== SubroutineSubprogram  ===========

  bool Pre(const SubroutineSubprogram &x) { 
    TRACE_CALL() ;
    return true ; 
  }

  void Post(const SubroutineSubprogram &x) { 
    TRACE_CALL() ;
  }

  // =========== Module =========== 

  bool Pre(const Module &x) { 
    TRACE_CALL() ;
    return true ; 
  }

  void Post(const Module &x) { 
    TRACE_CALL() ;
  }

  // =========== BlockData =========== 

  bool Pre(const BlockData &x) { 
    TRACE_CALL() ;
    return true ; 
  }

  void Post(const BlockData &x) { 
    TRACE_CALL() ;
  }


  // =========== FunctionStmt =========== 

  bool Pre(const FunctionStmt &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 
    (void) sema ; 
    (void) ConsumeLabel(x) ;
    return true ; 
  }

  void Post(const FunctionStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndFunctionStmt =========== 

  bool Pre(const EndFunctionStmt &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 
    (void) sema ; 
    (void) ConsumeLabel(x) ;
    return true ; 
  }

  void Post(const EndFunctionStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== TypeDeclarationStmt =========== 

  bool Pre(const TypeDeclarationStmt &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 
    (void) sema ; 
    (void) ConsumeLabel(x) ;
    return true ; 
  }

  void Post(const TypeDeclarationStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ImplicitStmt =========== 

  bool Pre(const ImplicitStmt &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 
    (void) sema ; 
    (void) ConsumeLabel(x) ;
    return true ; 
  }

  void Post(const ImplicitStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== PrintStmt =========== 

  bool Pre(const PrintStmt &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 
    (void) sema ; 
    (void) ConsumeLabel(x) ;
    return true ; 
  }

  void Post(const PrintStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== AssignmentStmt =========== 

  bool Pre(const AssignmentStmt &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 
    (void) sema ; 
    (void) ConsumeLabel(x) ;
    return true ; 
  }

  void Post(const AssignmentStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== AssignmentStmt =========== 

  bool Pre(const ProgramStmt &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 
    (void) sema ; 
    (void) ConsumeLabel(x) ;
    return true ; 
  }

  void Post(const ProgramStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== AssignmentStmt =========== 

  bool Pre(const EndProgramStmt &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 
    (void) sema ; 
    (void) ConsumeLabel(x) ;
    return true ; 
  }

  void Post(const EndProgramStmt &x) {     
    TRACE_CALL() ;
  }

#if 0
  
  // Do not remove: This is a skeleton for new node types
  // =========== XXX =========== 

  bool Pre(const XXX &x) { 
    TRACE_CALL() ;
    auto &sema = initSema(x); 
    (void) sema ; 
    // (void) ConsumeLabel(x) ;  
   return true ; 
  }

  void Post(const XXX &x) { 
    TRACE_CALL() ;
  }

#endif

public:
  
  void run(const ProgramUnit &p) {
    assert( NoLabelTable() ) ; 
    current_scope = unit_scope;
    Walk(p,*this) ;
    assert( current_scope == unit_scope ) ;
  }

} ;

}  // of namespace Fortran::parser 


void DoSemanticAnalysis( const psr::Program &all) 
{ 
  psr::Pass1 pass1 ;
  for (const psr::ProgramUnit &unit : all.v) {
    TRACE("===========================================================================================================");
    pass1.run(unit) ;
  } 
}

#endif
