//===--- FrontendOptions.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H
#define LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H

#include "clang/Frontend/CommandLineSourceLoc.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {

namespace frontend {
  enum ActionKind {
    ASTDump,                ///< Parse ASTs and dump them.
    ASTPrint,               ///< Parse ASTs and print them.
    ASTPrintXML,            ///< Parse ASTs and print them in XML.
    ASTView,                ///< Parse ASTs and view them in Graphviz.
    DumpRawTokens,          ///< Dump out raw tokens.
    DumpRecordLayouts,      ///< Dump record layout information.
    DumpTokens,             ///< Dump out preprocessed tokens.
    EmitAssembly,           ///< Emit a .s file.
    EmitBC,                 ///< Emit a .bc file.
    EmitHTML,               ///< Translate input source into HTML.
    EmitLLVM,               ///< Emit a .ll file.
    EmitLLVMOnly,           ///< Generate LLVM IR, but do not
    FixIt,                  ///< Parse and apply any fixits to the source.
    GeneratePCH,            ///< Generate pre-compiled header.
    GeneratePTH,            ///< Generate pre-tokenized header.
    HTMLTest,               ///< HTML displayer testing stuff.
    InheritanceView,        ///< View C++ inheritance for a specified class.
    ParseNoop,              ///< Parse with noop callbacks.
    ParsePrintCallbacks,    ///< Parse and print each callback.
    ParseSyntaxOnly,        ///< Parse and perform semantic analysis.
    PrintDeclContext,       ///< Print DeclContext and their Decls.
    PrintPreprocessedInput, ///< -E mode.
    RewriteBlocks,          ///< ObjC->C Rewriter for Blocks.
    RewriteMacros,          ///< Expand macros but not #includes.
    RewriteObjC,            ///< ObjC->C Rewriter.
    RewriteTest,            ///< Rewriter playground
    RunAnalysis,            ///< Run one or more source code analyses.
    RunPreprocessorOnly     ///< Just lex, no output.
  };
}

/// FrontendOptions - Options for controlling the behavior of the frontend.
class FrontendOptions {
public:
  enum InputKind {
    IK_None,
    IK_Asm,
    IK_C,
    IK_CXX,
    IK_ObjC,
    IK_ObjCXX,
    IK_PreprocessedC,
    IK_PreprocessedCXX,
    IK_PreprocessedObjC,
    IK_PreprocessedObjCXX,
    IK_OpenCL,
    IK_AST
  };

  unsigned DebugCodeCompletionPrinter : 1; ///< Use the debug printer for code
                                           /// completion results.
  unsigned DisableFree : 1;                ///< Disable memory freeing on exit.
  unsigned EmptyInputOnly : 1;             ///< Force input files to be treated
                                           /// as if they were empty, for timing
                                           /// the frontend startup.
  unsigned RelocatablePCH : 1;             ///< When generating PCH files,
                                           /// instruct the PCH writer to create
                                           /// relocatable PCH files.
  unsigned ShowMacrosInCodeCompletion : 1; ///< Show macros in code completion
                                           /// results.
  unsigned ShowStats : 1;                  ///< Show frontend performance
                                           /// metrics and statistics.
  unsigned ShowTimers : 1;                 ///< Show timers for individual
                                           /// actions.

  /// If given, the name of the target triple to compile for. If not given the
  /// target will be selected to match the host.
  std::string TargetTriple;

  /// If given, the name of the target ABI to use.
  std::string TargetABI;

  /// The input files and their types.
  std::vector<std::pair<InputKind, std::string> > Inputs;

  /// The output file, if any.
  std::string OutputFile;

  /// If given, the name for a C++ class to view the inheritance of.
  std::string ViewClassInheritance;

  /// A list of locations to apply fix-its at.
  std::vector<ParsedSourceLocation> FixItLocations;

  /// If given, enable code completion at the provided location.
  ParsedSourceLocation CodeCompletionAt;

  /// The frontend action to perform.
  frontend::ActionKind ProgramAction;

public:
  FrontendOptions() {
    DebugCodeCompletionPrinter = 0;
    DisableFree = 0;
    EmptyInputOnly = 0;
    ProgramAction = frontend::ParseSyntaxOnly;
    RelocatablePCH = 0;
    ShowMacrosInCodeCompletion = 0;
    ShowStats = 0;
    ShowTimers = 0;
  }

  /// getInputKindForExtension - Return the appropriate input kind for a file
  /// extension. For example, "c" would return IK_C.
  ///
  /// \return The input kind for the extension, or IK_None if the extension is
  /// not recognized.
  static InputKind getInputKindForExtension(llvm::StringRef Extension);
};

}  // end namespace clang

#endif
