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

#include <iosfwd>

namespace llvm { namespace sys { class Path; }}

namespace clang {

class ASTConsumer;
class Diagnostic;
class FileManager;
struct LangOptions;

ASTConsumer *CreateASTPrinter(std::ostream* OS = NULL);

ASTConsumer *CreateASTDumper();

ASTConsumer *CreateASTViewer();

ASTConsumer *CreateCFGDumper(bool ViewGraphs = false);

ASTConsumer *CreateLiveVarAnalyzer();

ASTConsumer *CreateDeadStoreChecker(Diagnostic &Diags);

ASTConsumer *CreateUnitValsChecker(Diagnostic &Diags);
  
ASTConsumer *CreateGRConstants();

ASTConsumer *CreateLLVMEmitter(Diagnostic &Diags, const LangOptions &Features);

ASTConsumer *CreateBCWriter(const std::string& InFile,
                            const std::string& OutFile,
                            Diagnostic &Diags,
                            const LangOptions &LOpts);

ASTConsumer *CreateCodeRewriterTest(Diagnostic &Diags);

ASTConsumer *CreateSerializationTest(Diagnostic &Diags,
                                     FileManager& FMgr, 
                                     const LangOptions &LOpts);
  
ASTConsumer *CreateASTSerializer(const std::string& InFile,
                                 const std::string& EmitDir,
                                 Diagnostic &Diags,
                                 const LangOptions &LOpts);

} // end clang namespace

#endif
