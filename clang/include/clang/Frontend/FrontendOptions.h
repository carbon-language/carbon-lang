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
#include "clang/Frontend/FrontendAction.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {

namespace frontend {
  enum ActionKind {
    ASTDump,                ///< Parse ASTs and dump them.
    ASTDumpXML,             ///< Parse ASTs and dump them in XML.
    ASTPrint,               ///< Parse ASTs and print them.
    ASTView,                ///< Parse ASTs and view them in Graphviz.
    CreateModule,           ///< Create module definition
    DumpRawTokens,          ///< Dump out raw tokens.
    DumpTokens,             ///< Dump out preprocessed tokens.
    EmitAssembly,           ///< Emit a .s file.
    EmitBC,                 ///< Emit a .bc file.
    EmitHTML,               ///< Translate input source into HTML.
    EmitLLVM,               ///< Emit a .ll file.
    EmitLLVMOnly,           ///< Generate LLVM IR, but do not emit anything.
    EmitCodeGenOnly,        ///< Generate machine code, but don't emit anything.
    EmitObj,                ///< Emit a .o file.
    FixIt,                  ///< Parse and apply any fixits to the source.
    GenerateModule,         ///< Generate pre-compiled module.
    GeneratePCH,            ///< Generate pre-compiled header.
    GeneratePTH,            ///< Generate pre-tokenized header.
    InitOnly,               ///< Only execute frontend initialization.
    ParseSyntaxOnly,        ///< Parse and perform semantic analysis.
    PluginAction,           ///< Run a plugin action, \see ActionName.
    PrintDeclContext,       ///< Print DeclContext and their Decls.
    PrintPreamble,          ///< Print the "preamble" of the input file
    PrintPreprocessedInput, ///< -E mode.
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
  unsigned DisableFree : 1;                ///< Disable memory freeing on exit.
  unsigned RelocatablePCH : 1;             ///< When generating PCH files,
                                           /// instruct the AST writer to create
                                           /// relocatable PCH files.
  unsigned ShowHelp : 1;                   ///< Show the -help text.
  unsigned ShowMacrosInCodeCompletion : 1; ///< Show macros in code completion
                                           /// results.
  unsigned ShowCodePatternsInCodeCompletion : 1; ///< Show code patterns in code
                                                 /// completion results.
  unsigned ShowGlobalSymbolsInCodeCompletion : 1; ///< Show top-level decls in
                                                  /// code completion results.
  unsigned ShowStats : 1;                  ///< Show frontend performance
                                           /// metrics and statistics.
  unsigned ShowTimers : 1;                 ///< Show timers for individual
                                           /// actions.
  unsigned ShowVersion : 1;                ///< Show the -version text.
  unsigned FixWhatYouCan : 1;              ///< Apply fixes even if there are
                                           /// unfixable errors.
  unsigned ARCMTMigrateEmitARCErrors : 1;  /// Emit ARC errors even if the
                                           /// migrator can fix them

  enum {
    ARCMT_None,
    ARCMT_Check,
    ARCMT_Modify,
    ARCMT_Migrate
  } ARCMTAction;

  std::string ARCMTMigrateDir;
  std::string ARCMTMigrateReportOut;

  /// The input files and their types.
  std::vector<std::pair<InputKind, std::string> > Inputs;

  /// The output file, if any.
  std::string OutputFile;

  /// If given, the new suffix for fix-it rewritten files.
  std::string FixItSuffix;

  /// If given, enable code completion at the provided location.
  ParsedSourceLocation CodeCompletionAt;

  /// The frontend action to perform.
  frontend::ActionKind ProgramAction;

  /// The name of the action to run when using a plugin action.
  std::string ActionName;

  /// Args to pass to the plugin
  std::vector<std::string> PluginArgs;

  /// The list of plugin actions to run in addition to the normal action.
  std::vector<std::string> AddPluginActions;

  /// Args to pass to the additional plugins
  std::vector<std::vector<std::string> > AddPluginArgs;

  /// The list of plugins to load.
  std::vector<std::string> Plugins;

  /// \brief The list of AST files to merge.
  std::vector<std::string> ASTMergeFiles;

  /// \brief The list of modules to import.
  std::vector<std::string> Modules;

  /// \brief A list of arguments to forward to LLVM's option processing; this
  /// should only be used for debugging and experimental features.
  std::vector<std::string> LLVMArgs;

public:
  FrontendOptions() {
    DisableFree = 0;
    ProgramAction = frontend::ParseSyntaxOnly;
    ActionName = "";
    RelocatablePCH = 0;
    ShowHelp = 0;
    ShowMacrosInCodeCompletion = 0;
    ShowCodePatternsInCodeCompletion = 0;
    ShowGlobalSymbolsInCodeCompletion = 1;
    ShowStats = 0;
    ShowTimers = 0;
    ShowVersion = 0;
    ARCMTAction = ARCMT_None;
    ARCMTMigrateEmitARCErrors = 0;
  }

  /// getInputKindForExtension - Return the appropriate input kind for a file
  /// extension. For example, "c" would return IK_C.
  ///
  /// \return The input kind for the extension, or IK_None if the extension is
  /// not recognized.
  static InputKind getInputKindForExtension(StringRef Extension);
};

}  // end namespace clang

#endif
