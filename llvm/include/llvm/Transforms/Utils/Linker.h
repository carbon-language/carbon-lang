//===- llvm/Transforms/Linker.h - Module Linker Interface --------*- C++ -*--=//
//
// This file defines the interface to the module linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMATIONS_LINKER_H
#define LLVM_TRANSFORMATIONS_LINKER_H

#include <string>
class Module;
class Type;
class Value;


// LinkModules - This function links two modules together, with the resulting
// left module modified to be the composite of the two input modules.  If an
// error occurs, true is returned and ErrorMsg (if not null) is set to indicate
// the problem.
//
bool    LinkModules(Module *Dest, const Module *Src, string *ErrorMsg = 0);


// MangleTypeName - Implement a consistent name-mangling scheme for
//                  a given type.
// 
string	MangleTypeName(const Type* type);


// MangleName - implement a consistent name-mangling scheme for all
// externally visible (i.e., global) objects.
// privateName should be unique within the module.
// 
string  MangleName(const string& privateName, const Value* V);


#endif
