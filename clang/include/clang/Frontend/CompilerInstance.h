//===-- CompilerInstance.h - Clang Compiler Instance ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_COMPILERINSTANCE_H_
#define LLVM_CLANG_FRONTEND_COMPILERINSTANCE_H_

#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/OwningPtr.h"
#include <cassert>
#include <list>
#include <string>

namespace llvm {
class LLVMContext;
class raw_ostream;
class raw_fd_ostream;
}

namespace clang {
class ASTContext;
class ASTConsumer;
class CodeCompleteConsumer;
class Diagnostic;
class DiagnosticClient;
class ExternalASTSource;
class FileManager;
class Preprocessor;
class Source;
class SourceManager;
class TargetInfo;

/// CompilerInstance - Helper class for managing a single instance of the Clang
/// compiler.
///
/// The CompilerInstance serves two purposes:
///  (1) It manages the various objects which are necessary to run the compiler,
///      for example the preprocessor, the target information, and the AST
///      context.
///  (2) It provides utility routines for constructing and manipulating the
///      common Clang objects.
///
/// The compiler instance generally owns the instance of all the objects that it
/// manages. However, clients can still share objects by manually setting the
/// object and retaking ownership prior to destroying the CompilerInstance.
///
/// The compiler instance is intended to simplify clients, but not to lock them
/// in to the compiler instance for everything. When possible, utility functions
/// come in two forms; a short form that reuses the CompilerInstance objects,
/// and a long form that takes explicit instances of any required objects.
class CompilerInstance {
  /// The LLVM context used for this instance.
  llvm::LLVMContext *LLVMContext;
  bool OwnsLLVMContext;

  /// The options used in this compiler instance.
  CompilerInvocation Invocation;

  /// The diagnostics engine instance.
  llvm::OwningPtr<Diagnostic> Diagnostics;

  /// The diagnostics client instance.
  llvm::OwningPtr<DiagnosticClient> DiagClient;

  /// The target being compiled for.
  llvm::OwningPtr<TargetInfo> Target;

  /// The file manager.
  llvm::OwningPtr<FileManager> FileMgr;

  /// The source manager.
  llvm::OwningPtr<SourceManager> SourceMgr;

  /// The preprocessor.
  llvm::OwningPtr<Preprocessor> PP;

  /// The AST context.
  llvm::OwningPtr<ASTContext> Context;

  /// The AST consumer.
  llvm::OwningPtr<ASTConsumer> Consumer;

  /// The code completion consumer.
  llvm::OwningPtr<CodeCompleteConsumer> CompletionConsumer;

  /// The list of active output files.
  std::list< std::pair<std::string, llvm::raw_ostream*> > OutputFiles;

public:
  /// Create a new compiler instance with the given LLVM context, optionally
  /// taking ownership of it.
  CompilerInstance(llvm::LLVMContext *_LLVMContext = 0,
                   bool _OwnsLLVMContext = true);
  ~CompilerInstance();

  /// @name LLVM Context
  /// {

  bool hasLLVMContext() const { return LLVMContext != 0; }

  llvm::LLVMContext &getLLVMContext() const {
    assert(LLVMContext && "Compiler instance has no LLVM context!");
    return *LLVMContext;
  }

  /// setLLVMContext - Replace the current LLVM context and take ownership of
  /// \arg Value.
  void setLLVMContext(llvm::LLVMContext *Value, bool TakeOwnership = true) {
    LLVMContext = Value;
    OwnsLLVMContext = TakeOwnership;
  }

  /// }
  /// @name Compiler Invocation and Options
  /// {

  CompilerInvocation &getInvocation() { return Invocation; }
  const CompilerInvocation &getInvocation() const { return Invocation; }
  void setInvocation(const CompilerInvocation &Value) { Invocation = Value; }

  /// }
  /// @name Forwarding Methods
  /// {

  AnalyzerOptions &getAnalyzerOpts() {
    return Invocation.getAnalyzerOpts();
  }
  const AnalyzerOptions &getAnalyzerOpts() const {
    return Invocation.getAnalyzerOpts();
  }

  CodeGenOptions &getCodeGenOpts() {
    return Invocation.getCodeGenOpts();
  }
  const CodeGenOptions &getCodeGenOpts() const {
    return Invocation.getCodeGenOpts();
  }

  DependencyOutputOptions &getDependencyOutputOpts() {
    return Invocation.getDependencyOutputOpts();
  }
  const DependencyOutputOptions &getDependencyOutputOpts() const {
    return Invocation.getDependencyOutputOpts();
  }

  DiagnosticOptions &getDiagnosticOpts() {
    return Invocation.getDiagnosticOpts();
  }
  const DiagnosticOptions &getDiagnosticOpts() const {
    return Invocation.getDiagnosticOpts();
  }

  FrontendOptions &getFrontendOpts() {
    return Invocation.getFrontendOpts();
  }
  const FrontendOptions &getFrontendOpts() const {
    return Invocation.getFrontendOpts();
  }

  HeaderSearchOptions &getHeaderSearchOpts() {
    return Invocation.getHeaderSearchOpts();
  }
  const HeaderSearchOptions &getHeaderSearchOpts() const {
    return Invocation.getHeaderSearchOpts();
  }

  LangOptions &getLangOpts() {
    return Invocation.getLangOpts();
  }
  const LangOptions &getLangOpts() const {
    return Invocation.getLangOpts();
  }

  PreprocessorOptions &getPreprocessorOpts() {
    return Invocation.getPreprocessorOpts();
  }
  const PreprocessorOptions &getPreprocessorOpts() const {
    return Invocation.getPreprocessorOpts();
  }

  PreprocessorOutputOptions &getPreprocessorOutputOpts() {
    return Invocation.getPreprocessorOutputOpts();
  }
  const PreprocessorOutputOptions &getPreprocessorOutputOpts() const {
    return Invocation.getPreprocessorOutputOpts();
  }

  /// }
  /// @name Diagnostics Engine
  /// {

  bool hasDiagnostics() const { return Diagnostics != 0; }

  Diagnostic &getDiagnostics() const {
    assert(Diagnostics && "Compiler instance has no diagnostics!");
    return *Diagnostics;
  }

  /// takeDiagnostics - Remove the current diagnostics engine and give ownership
  /// to the caller.
  Diagnostic *takeDiagnostics() { return Diagnostics.take(); }

  /// setDiagnostics - Replace the current diagnostics engine; the compiler
  /// instance takes ownership of \arg Value.
  void setDiagnostics(Diagnostic *Value);

  DiagnosticClient &getDiagnosticClient() const;

  /// takeDiagnosticClient - Remove the current diagnostics client and give
  /// ownership to the caller.
  DiagnosticClient *takeDiagnosticClient() { return DiagClient.take(); }

  /// setDiagnosticClient - Replace the current diagnostics client; the compiler
  /// instance takes ownership of \arg Value.
  void setDiagnosticClient(DiagnosticClient *Value);

  /// }
  /// @name Target Info
  /// {

  bool hasTarget() const { return Target != 0; }

  TargetInfo &getTarget() const {
    assert(Target && "Compiler instance has no target!");
    return *Target;
  }

  /// takeTarget - Remove the current diagnostics engine and give ownership
  /// to the caller.
  TargetInfo *takeTarget() { return Target.take(); }

  /// setTarget - Replace the current diagnostics engine; the compiler
  /// instance takes ownership of \arg Value.
  void setTarget(TargetInfo *Value);

  /// }
  /// @name File Manager
  /// {

  bool hasFileManager() const { return FileMgr != 0; }

  FileManager &getFileManager() const {
    assert(FileMgr && "Compiler instance has no file manager!");
    return *FileMgr;
  }

  /// takeFileManager - Remove the current file manager and give ownership to
  /// the caller.
  FileManager *takeFileManager() { return FileMgr.take(); }

  /// setFileManager - Replace the current file manager; the compiler instance
  /// takes ownership of \arg Value.
  void setFileManager(FileManager *Value);

  /// }
  /// @name Source Manager
  /// {

  bool hasSourceManager() const { return SourceMgr != 0; }

  SourceManager &getSourceManager() const {
    assert(SourceMgr && "Compiler instance has no source manager!");
    return *SourceMgr;
  }

  /// takeSourceManager - Remove the current source manager and give ownership
  /// to the caller.
  SourceManager *takeSourceManager() { return SourceMgr.take(); }

  /// setSourceManager - Replace the current source manager; the compiler
  /// instance takes ownership of \arg Value.
  void setSourceManager(SourceManager *Value);

  /// }
  /// @name Preprocessor
  /// {

  bool hasPreprocessor() const { return PP != 0; }

  Preprocessor &getPreprocessor() const {
    assert(PP && "Compiler instance has no preprocessor!");
    return *PP;
  }

  /// takePreprocessor - Remove the current preprocessor and give ownership to
  /// the caller.
  Preprocessor *takePreprocessor() { return PP.take(); }

  /// setPreprocessor - Replace the current preprocessor; the compiler instance
  /// takes ownership of \arg Value.
  void setPreprocessor(Preprocessor *Value);

  /// }
  /// @name ASTContext
  /// {

  bool hasASTContext() const { return Context != 0; }

  ASTContext &getASTContext() const {
    assert(Context && "Compiler instance has no AST context!");
    return *Context;
  }

  /// takeASTContext - Remove the current AST context and give ownership to the
  /// caller.
  ASTContext *takeASTContext() { return Context.take(); }

  /// setASTContext - Replace the current AST context; the compiler instance
  /// takes ownership of \arg Value.
  void setASTContext(ASTContext *Value);

  /// }
  /// @name ASTConsumer
  /// {

  bool hasASTConsumer() const { return Consumer != 0; }

  ASTConsumer &getASTConsumer() const {
    assert(Consumer && "Compiler instance has no AST consumer!");
    return *Consumer;
  }

  /// takeASTConsumer - Remove the current AST consumer and give ownership to
  /// the caller.
  ASTConsumer *takeASTConsumer() { return Consumer.take(); }

  /// setASTConsumer - Replace the current AST consumer; the compiler instance
  /// takes ownership of \arg Value.
  void setASTConsumer(ASTConsumer *Value);

  /// }
  /// @name Code Completion
  /// {

  bool hasCodeCompletionConsumer() const { return CompletionConsumer != 0; }

  CodeCompleteConsumer &getCodeCompletionConsumer() const {
    assert(CompletionConsumer &&
           "Compiler instance has no code completion consumer!");
    return *CompletionConsumer;
  }

  /// takeCodeCompletionConsumer - Remove the current code completion consumer
  /// and give ownership to the caller.
  CodeCompleteConsumer *takeCodeCompletionConsumer() {
    return CompletionConsumer.take();
  }

  /// setCodeCompletionConsumer - Replace the current code completion consumer;
  /// the compiler instance takes ownership of \arg Value.
  void setCodeCompletionConsumer(CodeCompleteConsumer *Value);

  /// }
  /// @name Output Files
  /// {

  /// getOutputFileList - Get the list of (path, output stream) pairs of output
  /// files; the path may be empty but the stream will always be non-null.
  const std::list< std::pair<std::string,
                             llvm::raw_ostream*> > &getOutputFileList() const;

  /// addOutputFile - Add an output file onto the list of tracked output files.
  ///
  /// \param Path - The path to the output file, or empty.
  /// \param OS - The output stream, which should be non-null.
  void addOutputFile(llvm::StringRef Path, llvm::raw_ostream *OS);

  /// ClearOutputFiles - Clear the output file list, destroying the contained
  /// output streams.
  ///
  /// \param EraseFiles - If true, attempt to erase the files from disk.
  void ClearOutputFiles(bool EraseFiles);

  /// }
  /// @name Construction Utility Methods
  /// {

  /// Create the diagnostics engine using the invocation's diagnostic options
  /// and replace any existing one with it.
  ///
  /// Note that this routine also replaces the diagnostic client.
  void createDiagnostics(int Argc, char **Argv);

  /// Create a Diagnostic object with a the TextDiagnosticPrinter.
  ///
  /// The \arg Argc and \arg Argv arguments are used only for logging purposes,
  /// when the diagnostic options indicate that the compiler should output
  /// logging information.
  ///
  /// Note that this creates an unowned DiagnosticClient, if using directly the
  /// caller is responsible for releaseing the returned Diagnostic's client
  /// eventually.
  ///
  /// \return The new object on success, or null on failure.
  static Diagnostic *createDiagnostics(const DiagnosticOptions &Opts,
                                       int Argc, char **Argv);

  /// Create the file manager and replace any existing one with it.
  void createFileManager();

  /// Create the source manager and replace any existing one with it.
  void createSourceManager();

  /// Create the preprocessor, using the invocation, file, and source managers,
  /// and replace any existing one with it.
  void createPreprocessor();

  /// Create a Preprocessor object.
  ///
  /// Note that this also creates a new HeaderSearch object which will be owned
  /// by the resulting Preprocessor.
  ///
  /// \return The new object on success, or null on failure.
  static Preprocessor *createPreprocessor(Diagnostic &, const LangOptions &,
                                          const PreprocessorOptions &,
                                          const HeaderSearchOptions &,
                                          const DependencyOutputOptions &,
                                          const TargetInfo &,
                                          SourceManager &, FileManager &);

  /// Create the AST context.
  void createASTContext();

  /// Create an external AST source to read a PCH file and attach it to the AST
  /// context.
  void createPCHExternalASTSource(llvm::StringRef Path);

  /// Create an external AST source to read a PCH file.
  ///
  /// \return - The new object on success, or null on failure.
  static ExternalASTSource *
  createPCHExternalASTSource(llvm::StringRef Path, const std::string &Sysroot,
                             Preprocessor &PP, ASTContext &Context);

  /// Create a code completion consumer using the invocation; note that this
  /// will cause the source manager to truncate the input source file at the
  /// completion point.
  void createCodeCompletionConsumer();

  /// Create a code completion consumer to print code completion results, at
  /// \arg Filename, \arg Line, and \arg Column, to the given output stream \arg
  /// OS.
  static CodeCompleteConsumer *
  createCodeCompletionConsumer(Preprocessor &PP, const std::string &Filename,
                               unsigned Line, unsigned Column,
                               bool UseDebugPrinter, bool ShowMacros,
                               llvm::raw_ostream &OS);

  /// Create the default output file (from the invocation's options) and add it
  /// to the list of tracked output files.
  llvm::raw_fd_ostream *
  createDefaultOutputFile(bool Binary = true, llvm::StringRef BaseInput = "",
                          llvm::StringRef Extension = "");

  /// Create a new output file and add it to the list of tracked output files,
  /// optionally deriving the output path name.
  llvm::raw_fd_ostream *
  createOutputFile(llvm::StringRef OutputPath, bool Binary = true,
                   llvm::StringRef BaseInput = "",
                   llvm::StringRef Extension = "");

  /// Create a new output file, optionally deriving the output path name.
  ///
  /// If \arg OutputPath is empty, then createOutputFile will derive an output
  /// path location as \arg BaseInput, with any suffix removed, and \arg
  /// Extension appended.
  ///
  /// \param OutputPath - If given, the path to the output file.
  /// \param Error [out] - On failure, the error message.
  /// \param BaseInput - If \arg OutputPath is empty, the input path name to use
  /// for deriving the output path.
  /// \param Extension - The extension to use for derived output names.
  /// \param Binary - The mode to open the file in.
  /// \param ResultPathName [out] - If given, the result path name will be
  /// stored here on success.
  static llvm::raw_fd_ostream *
  createOutputFile(llvm::StringRef OutputPath, std::string &Error,
                   bool Binary = true, llvm::StringRef BaseInput = "",
                   llvm::StringRef Extension = "",
                   std::string *ResultPathName = 0);

  /// }
};

} // end namespace clang

#endif
