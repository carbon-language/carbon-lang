//===--- ASTConsumers.h - ASTConsumer implementations -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AST Consumers.
//
//===----------------------------------------------------------------------===//

#ifndef DRIVER_ASTCONSUMERS_H
#define DRIVER_ASTCONSUMERS_H

#include <string>

namespace llvm {
  class raw_ostream;
  class Module;
  class LLVMContext;
  namespace sys { class Path; }
}
namespace clang {

class ASTConsumer;
class CodeGenOptions;
class Diagnostic;
class FileManager;
class LangOptions;
class Preprocessor;
class TargetOptions;

// AST pretty-printer: prints out the AST in a format that is close to the
// original C code.  The output is intended to be in a format such that
// clang could re-parse the output back into the same AST, but the
// implementation is still incomplete.
ASTConsumer *CreateASTPrinter(llvm::raw_ostream *OS);

// AST XML-printer: prints out the AST in a XML format
// The output is intended to be in a format such that
// clang or any other tool could re-parse the output back into the same AST,
// but the implementation is still incomplete.
ASTConsumer *CreateASTPrinterXML(llvm::raw_ostream *OS);

// AST dumper: dumps the raw AST in human-readable form to stderr; this is
// intended for debugging.
ASTConsumer *CreateASTDumper();

// Graphical AST viewer: for each function definition, creates a graph of
// the AST and displays it with the graph viewer "dotty".  Also outputs
// function declarations to stderr.
ASTConsumer *CreateASTViewer();

// DeclContext printer: prints out the DeclContext tree in human-readable form
// to stderr; this is intended for debugging.
ASTConsumer *CreateDeclContextPrinter();

// Inheritance viewer: for C++ code, creates a graph of the inheritance
// tree for the given class and displays it with "dotty".
ASTConsumer *CreateInheritanceViewer(const std::string& clsname);

} // end clang namespace

#endif
