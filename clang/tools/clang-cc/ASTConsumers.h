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

#include "llvm/Support/raw_ostream.h"
#include <string>
#include <iosfwd>

namespace llvm {
  class Module;
  namespace sys { class Path; }
}
namespace clang {

class ASTConsumer;
class Diagnostic;
class FileManager;
class Preprocessor;
class PreprocessorFactory;
struct CompileOptions;
class LangOptions;

ASTConsumer *CreateASTPrinter(llvm::raw_ostream* OS = NULL);

ASTConsumer *CreateASTDumper(bool FullDump);

ASTConsumer *CreateASTViewer();

ASTConsumer *CreateDeclContextPrinter();

ASTConsumer *CreateCodeRewriterTest(const std::string& InFile,
                                    const std::string& OutFile,
                                    Diagnostic &Diags,
                                    const LangOptions &LOpts);
  
enum BackendAction {
  Backend_EmitAssembly,
  Backend_EmitBC,
  Backend_EmitLL,
  Backend_EmitNothing
};
ASTConsumer *CreateBackendConsumer(BackendAction Action,
                                   Diagnostic &Diags,
                                   const LangOptions &Features,
                                   const CompileOptions &CompileOpts,
                                   const std::string &InFile,
                                   const std::string &OutFile);

ASTConsumer* CreateHTMLPrinter(const std::string &OutFile, Diagnostic &D,
                               Preprocessor *PP, PreprocessorFactory *PPF);

ASTConsumer *CreatePCHGenerator(const Preprocessor &PP,
                                const std::string &OutFile);

ASTConsumer *CreateBlockRewriter(const std::string &InFile,
                                 const std::string &OutFile,
                                 Diagnostic &Diags,
                                 const LangOptions &LangOpts);
  
ASTConsumer *CreateInheritanceViewer(const std::string& clsname);

ASTConsumer* CreateAnalysisConsumer(Diagnostic &diags, Preprocessor *pp,
                                    PreprocessorFactory *ppf,
                                    const LangOptions &lopts,
                                    const std::string &output);

} // end clang namespace

#endif
