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
class Diagnostic;
class FileManager;
class Preprocessor;
class CompileOptions;
class LangOptions;

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

// RecordLayout dumper: prints out the record layout information for all records
// in the translation unit; this is intended for debugging.
ASTConsumer *CreateRecordLayoutDumper();

// ObjC rewriter: attempts tp rewrite ObjC constructs into pure C code.
// This is considered experimental, and only works with Apple's ObjC runtime.
ASTConsumer *CreateObjCRewriter(const std::string &InFile,
                                llvm::raw_ostream *OS,
                                Diagnostic &Diags,
                                const LangOptions &LOpts,
                                bool SilenceRewriteMacroWarning);

// LLVM code generator: uses the code generation backend to generate LLVM
// assembly. This runs optimizations depending on the CompileOptions
// parameter. The output depends on the Action parameter.
enum BackendAction {
  Backend_EmitAssembly,  // Emit native assembly
  Backend_EmitBC,        // Emit LLVM bitcode file
  Backend_EmitLL,        // Emit human-readable LLVM assembly
  Backend_EmitNothing    // Don't emit anything (benchmarking mode)
};
ASTConsumer *CreateBackendConsumer(BackendAction Action,
                                   Diagnostic &Diags,
                                   const LangOptions &Features,
                                   const CompileOptions &CompileOpts,
                                   const std::string &ModuleID,
                                   llvm::raw_ostream *OS,
                                   llvm::LLVMContext& C);

/// CreateHTMLPrinter - Create an AST consumer which rewrites source code to
/// HTML with syntax highlighting suitable for viewing in a web-browser.
ASTConsumer *CreateHTMLPrinter(llvm::raw_ostream *OS, Preprocessor &PP,
                               bool SyntaxHighlight = true,
                               bool HighlightMacros = true);

// PCH generator: generates a precompiled header file; this file can be
// used later with the PCHReader (clang-cc option -include-pch)
// to speed up compile times.
ASTConsumer *CreatePCHGenerator(const Preprocessor &PP,
                                llvm::raw_ostream *OS,
                                const char *isysroot = 0);

// Block rewriter: rewrites code using the Apple blocks extension to pure
// C code.  Output is always sent to stdout.
ASTConsumer *CreateBlockRewriter(const std::string &InFile,
                                 Diagnostic &Diags,
                                 const LangOptions &LangOpts);

// Inheritance viewer: for C++ code, creates a graph of the inheritance
// tree for the given class and displays it with "dotty".
ASTConsumer *CreateInheritanceViewer(const std::string& clsname);

} // end clang namespace

#endif
