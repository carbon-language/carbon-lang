#include "ParseTreeDump.h"

#include <string>
#include <cstring>
#include <iostream>
#include <iomanip>

#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>

namespace Fortran::semantics {

std::string DemangleCxxName(const char* name) {
  std::string result; 
  int status = -4; // some arbitrary value to eliminate the compiler warning
  char * output =  abi::__cxa_demangle(name, NULL, NULL, &status);
  if ( status==0 ) {
    std::string result(output);
    free(output) ;
    return result ;
  } else {
    return name ;
  } 
}

} // of namespace  Fortran::semantics

#else

// Nothing if not G++ 
namespace Fortran::semantics {
std::string DemangleCxxName(const char* name) {
  return name;
}
} 

#endif

namespace Fortran::parser {

bool ParseTreeDumper::startwith( const std::string str, const char *prefix) {
  size_t len = strlen(prefix) ;
  return (str.compare(0,len,prefix)==0) ;
} 

std::string ParseTreeDumper::cleanup(const std::string &name) {
  if ( startwith(name,"Fortran::parser::") )
    return name.substr(strlen("Fortran::parser::")) ;
  else
    return name;
}

// Perform all required instantiations  
FLANG_PARSE_TREE_DUMPER_INSTANTIATE_ALL()  

} // of namespace



