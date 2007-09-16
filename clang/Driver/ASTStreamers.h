//===--- ASTStreamers.h - ASTStreamer Drivers -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AST Streamers.
//
//===----------------------------------------------------------------------===//

#ifndef DRIVER_ASTSTREAMERS_H_
#define DRIVER_ASTSTREAMERS_H_

namespace clang {

class ASTConsumer;
class Diagnostic;

ASTConsumer *CreateASTPrinter();
ASTConsumer *CreateASTDumper();
ASTConsumer *CreateCFGDumper(bool ViewGraphs = false);
ASTConsumer *CreateLiveVarAnalyzer();
ASTConsumer *CreateDeadStoreChecker(Diagnostic &Diags);
ASTConsumer *CreateLLVMEmitter(Diagnostic &Diags);

} // end clang namespace

#endif
