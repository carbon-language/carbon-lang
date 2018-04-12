#ifndef FLANG_SEMA_PARSE_TREE_DUMP_H
#define FLANG_SEMA_PARSE_TREE_DUMP_H

#include "../parser/format-specification.h"
#include "../parser/idioms.h"
#include "../parser/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace Fortran::semantics {


std::string DemangleCxxName(const char* name) ;

template <typename T> std::string GetTypeName_base() {
  return DemangleCxxName( typeid(T).name() ) ;
}

template <typename T> std::string GetTypeName() {
  return GetTypeName_base< 
    typename std::remove_cv<
      typename std::remove_reference<        
        T
        >::type
      >::type
    > ();   
} 

// Make it usable on 
template <typename T> std::string GetTypeName(const T &x) {
  return GetTypeName<decltype(x)>() ;
} 

// Simplify the name of some types
 
#define FLANG_PARSER_RENAME_TYPE( TYPE, NAME ) \
  template <> inline std::string GetTypeName_base<TYPE>() { return NAME; }  

FLANG_PARSER_RENAME_TYPE( Fortran::parser::LoopBounds<Fortran::parser::ScalarIntConstantExpr> ,
             "LoopBounds<Expr>")

FLANG_PARSER_RENAME_TYPE( Fortran::parser::LoopBounds<Fortran::parser::ScalarIntExpr> ,
             "LoopBounds<Expr>")


} // end of namespace


namespace Fortran::parser {

//
// Dump the Parse Tree hiearchy of any node 'x' of the parse tree.
//
// ParseTreeDumper().run(x)
//

class ParseTreeDumper {
private:
  int indent;
  std::ostream &out ;  
  bool emptyline;
public:
  
  ParseTreeDumper(std::ostream &out_ = std::cerr) : indent(0) , out(out_) , emptyline(false) { }

private:

  static bool startwith( const std::string str, const char *prefix) ;
  static std::string cleanup(const std::string &name) ;


public:

  void out_indent() {
    for (int i=0;i<indent;i++) {
      out << "| " ;
    }
  }


  template <typename T> bool Pre(const T &x) { 
    if (emptyline ) {
      out_indent();
      emptyline = false ;
    }
    if ( UnionTrait<T> || WrapperTrait<T> ) {
      out << cleanup(Fortran::semantics::GetTypeName<decltype(x)>()) << " -> "  ;
      emptyline = false ;
    } else {
      out << cleanup(Fortran::semantics::GetTypeName<decltype(x)>())    ;
      out << "\n" ; 
      indent++ ;
      emptyline = true ;
    }    
    return true ;
  }
  
  template <typename T> void Post(const T &x) { 
    if ( UnionTrait<T> || WrapperTrait<T> ) {
      if (!emptyline) { 
        out << "\n" ; 
        emptyline = true ; 
      }
    } else {
      indent--;
    }
  }

  bool Pre(const parser::Name &x) { return Pre(x.ToString()); }

  bool Pre(const std::string &x) { 
    if (emptyline ) {
      out_indent();
      emptyline = false ;
    }    
    out << "Name = '" << x << "'\n";
    indent++ ;
    emptyline = true ;    
    return true ;
  }
  
  void Post(const std::string &x) { 
    indent--;
  }

  bool Pre(const std::int64_t &x) { 
    if (emptyline ) {
      out_indent();
      emptyline = false ;
    }    
    out << "int = '" << x << "'\n";
    indent++ ;
    emptyline = true ;    
    return true ;
  }
  
  void Post(const std::int64_t &x) { 
    indent--;
  }

  bool Pre(const std::uint64_t &x) { 
    if (emptyline ) {
      out_indent();
      emptyline = false ;
    }    
    out << "int = '" << x << "'\n";
    indent++ ;
    emptyline = true ;    
    return true ;
  }
  
  void Post(const std::uint64_t &x) { 
    indent--;
  }

  // A few types we want to ignore


  template <typename T> bool Pre(const Fortran::parser::Statement<T> &) { 
    return true;
  }

  template <typename T> void Post(const Fortran::parser::Statement<T> &) { 
  }

  template <typename T> bool Pre(const Fortran::parser::Indirection<T> &) { 
    return true;
  }

  template <typename T> void Post(const Fortran::parser::Indirection<T> &) { 
  }

  template <typename T> bool Pre(const Fortran::parser::Integer<T> &) { 
    return true;
  }

  template <typename T> void Post(const Fortran::parser::Integer<T> &) { 
  }


  template <typename T> bool Pre(const Fortran::parser::Scalar<T> &) { 
    return true;
  }

  template <typename T> void Post(const Fortran::parser::Scalar<T> &) { 
  }

  template <typename... A> bool Pre(const std::tuple<A...> &) { 
    return true;
  }

  template <typename... A> void Post(const std::tuple<A...> &) { 
  }

  template <typename... A> bool Pre(const std::variant<A...> &) { 
    return true;
  }
  
  template <typename... A> void Post(const std::variant<A...> &) { 
  }


public:
  
  
}; 


template <typename T>
void DumpTree(const T &x, std::ostream &out=std::cout )
{
  ParseTreeDumper dumper(out);
  Fortran::parser::Walk(x,dumper); 
}


} // of namespace 


namespace Fortran::parser {

// Provide a explicit instantiation for a few selected node types.
// The goal is not to provide the instanciation of all possible 
// types but to insure that a call to DumpTree will not cause
// the instanciation of thousands of types.
//
 

#define FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,TYPE) \
 MODE template void Walk(const TYPE&, Fortran::parser::ParseTreeDumper &);

#define FLANG_PARSE_TREE_DUMPER_INSTANTIATE_ALL(MODE) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ProgramUnit) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,SubroutineStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ProgramStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,FunctionStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ModuleStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,Expr) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ActionStmt) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,ExecutableConstruct) \
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,Block)\
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,DeclarationConstruct)\
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,SpecificationPart)\
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,OtherSpecificationStmt)\
  FLANG_PARSE_TREE_DUMPER_INSTANTIATE(MODE,SpecificationConstruct)\



FLANG_PARSE_TREE_DUMPER_INSTANTIATE_ALL(extern) 


} // of namespace 

#endif
