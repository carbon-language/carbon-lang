//===--- CodeGenFunction.h - Per-Function state for LLVM CodeGen ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-function state used for llvm translation. 
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_CODEGENFUNCTION_H
#define CODEGEN_CODEGENFUNCTION_H

namespace llvm {
  class Module;
namespace clang {
  class ASTContext;
  class FunctionDecl;
  
namespace CodeGen {
  class CodeGenModule;
  
/// CodeGenFunction - This class organizes the per-function state that is used
/// while generating LLVM code.
class CodeGenFunction {
  CodeGenModule &CGM;  // Per-module state.
public:
  CodeGenFunction(CodeGenModule &cgm) : CGM(cgm) {}
  
};
}  // end namespace CodeGen
}  // end namespace clang
}  // end namespace llvm

#endif
