//===--- ASTConsumers.h - ASTConsumer implementations -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AST Consumers.
//
//===----------------------------------------------------------------------===//

#ifndef DRIVER_ASTCONSUMERS_H
#define DRIVER_ASTCONSUMERS_H

#include <iosfwd>

namespace clang {

class ASTConsumer;
class Diagnostic;
struct LangOptions;

ASTConsumer *CreateASTPrinter(std::ostream* OS = NULL);
ASTConsumer *CreateASTDumper();
ASTConsumer *CreateASTViewer();
ASTConsumer *CreateCFGDumper(bool ViewGraphs = false);
ASTConsumer *CreateLiveVarAnalyzer();
ASTConsumer *CreateDeadStoreChecker(Diagnostic &Diags);
ASTConsumer *CreateUnitValsChecker(Diagnostic &Diags);
ASTConsumer *CreateLLVMEmitter(Diagnostic &Diags, const LangOptions &Features);
ASTConsumer *CreateCodeRewriterTest();
ASTConsumer *CreateSerializationTest();

} // end clang namespace

#endif
