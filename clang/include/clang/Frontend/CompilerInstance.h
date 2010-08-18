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
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/OwningPtr.h"
#include <cassert>
#include <list>
#include <string>

namespace llvm {
class LLVMContext;
class raw_ostream;
class raw_fd_ostream;
class Timer;
}

namespace clang {
class ASTContext;
class ASTConsumer;
class CodeCompleteConsumer;
class Diagnostic;
class DiagnosticClient;
class ExternalASTSource;
class FileManager;
class FrontendAction;
class ASTReader;
class Preprocessor;
class Sema;
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
  llvm::OwningPtr<llvm::LLVMContext> LLVMContext;

  /// The options used in this compiler instance.
  llvm::OwningPtr<CompilerInvocation> Invocation;

  /// The diagnostics engine instance.
  llvm::IntrusiveRefCntPtr<Diagnostic> Diagnostics;

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

  /// \brief The semantic analysis object.
  llvm::OwningPtr<Sema> TheSema;
  
  /// The frontend timer
  llvm::OwningPtr<llvm::Timer> FrontendTimer;

  /// The list of active output files.
  std::list< std::pair<std::string, llvm::raw_ostream*> > OutputFiles;

  void operator=(const CompilerInstance &);  // DO NOT IMPLEMENT
  CompilerInstance(const CompilerInstance&); // DO NOT IMPLEMENT
public:
  CompilerInstance();
  ~CompilerInstance();

  /// @name High-Level Operations
  /// {

  /// ExecuteAction - Execute the provided action against the compiler's
  /// CompilerInvocation object.
  ///
  /// This function makes the following assumptions:
  ///
  ///  - The invocation options should be initialized. This function does not
  ///    handle the '-help' or '-version' options, clients should handle those
  ///    directly.
  ///
  ///  - The diagnostics engine should have already been created by the client.
  ///
  ///  - No other CompilerInstance state should have been initialized (this is
  ///    an unchecked error).
  ///
  ///  - Clients should have initialized any LLVM target features that may be
  ///    required.
  ///
  ///  - Clients should eventually call llvm_shutdown() upon the completion of
  ///    this routine to ensure that any managed objects are properly destroyed.
  ///
  /// Note that this routine may write output to 'stderr'.
  ///
  /// \param Act - The action to execute.
  /// \return - True on success.
  //
  // FIXME: This function should take the stream to write any debugging /
  // verbose output to as an argument.
  //
  // FIXME: Eliminate the llvm_shutdown requirement, that should either be part
  // of the context or else not CompilerInstance specific.
  bool ExecuteAction(FrontendAction &Act);

  /// }
  /// @name LLVM Context
  /// {

  bool hasLLVMContext() const { return LLVMContext != 0; }

  llvm::LLVMContext &getLLVMContext() const {
    assert(LLVMContext && "Compiler instance has no LLVM context!");
    return *LLVMContext;
  }

  llvm::LLVMContext *takeLLVMContext() { return LLVMContext.take(); }

  /// setLLVMContext - Replace the current LLVM context and take ownership of
  /// \arg Value.
  void setLLVMContext(llvm::LLVMContext *Value);

  /// }
  /// @name Compiler Invocation and Options
  /// {

  bool hasInvocation() const { return Invocation != 0; }

  CompilerInvocation &getInvocation() {
    assert(Invocation && "Compiler instance has no invocation!");
    return *Invocation;
  }

  CompilerInvocation *takeInvocation() { return Invocation.take(); }

  /// setInvocation - Replace the current invocation; the compiler instance
  /// takes ownership of \arg Value.
  void setInvocation(CompilerInvocation *Value);

  /// }
  /// @name Forwarding Methods
  /// {

  AnalyzerOptions &getAnalyzerOpts() {
    return Invocation->getAnalyzerOpts();
  }
  const AnalyzerOptions &getAnalyzerOpts() const {
    return Invocation->getAnalyzerOpts();
  }

  CodeGenOptions &getCodeGenOpts() {
    return Invocation->getCodeGenOpts();
  }
  const CodeGenOptions &getCodeGenOpts() const {
    return Invocation->getCodeGenOpts();
  }

  DependencyOutputOptions &getDependencyOutputOpts() {
    return Invocation->getDependencyOutputOpts();
  }
  const DependencyOutputOptions &getDependencyOutputOpts() const {
    return Invocation->getDependencyOutputOpts();
  }

  DiagnosticOptions &getDiagnosticOpts() {
    return Invocation->getDiagnosticOpts();
  }
  const DiagnosticOptions &getDiagnosticOpts() const {
    return Invocation->getDiagnosticOpts();
  }

  FrontendOptions &getFrontendOpts() {
    return Invocation->getFrontendOpts();
  }
  const FrontendOptions &getFrontendOpts() const {
    return Invocation->getFrontendOpts();
  }

  HeaderSearchOptions &getHeaderSearchOpts() {
    return Invocation->getHeaderSearchOpts();
  }
  const HeaderSearchOptions &getHeaderSearchOpts() const {
    return Invocation->getHeaderSearchOpts();
  }

  LangOptions &getLangOpts() {
    return Invocation->getLangOpts();
  }
  const LangOptions &getLangOpts() const {
    return Invocation->getLangOpts();
  }

  PreprocessorOptions &getPreprocessorOpts() {
    return Invocation->getPreprocessorOpts();
  }
  const PreprocessorOptions &getPreprocessorOpts() const {
    return Invocation->getPreprocessorOpts();
  }

  PreprocessorOutputOptions &getPreprocessorOutputOpts() {
    return Invocation->getPreprocessorOutputOpts();
  }
  const PreprocessorOutputOptions &getPreprocessorOutputOpts() const {
    return Invocation->getPreprocessorOutputOpts();
  }

  TargetOptions &getTargetOpts() {
    return Invocation->getTargetOpts();
  }
  const TargetOptions &getTargetOpts() const {
    return Invocation->getTargetOpts();
  }

  /// }
  /// @name Diagnostics Engine
  /// {

  bool hasDiagnostics() const { return Diagnostics != 0; }

  Diagnostic &getDiagnostics() const {
    assert(Diagnostics && "Compiler instance has no diagnostics!");
    return *Diagnostics;
  }

  /// setDiagnostics - Replace the current diagnostics engine; the compiler
  /// instance takes ownership of \arg Value.
  void setDiagnostics(Diagnostic *Value);

  DiagnosticClient &getDiagnosticClient() const {
    assert(Diagnostics && Diagnostics->getClient() && 
           "Compiler instance has no diagnostic client!");
    return *Diagnostics->getClient();
  }

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

  /// \brief Replace the current Sema; the compiler instance takes ownership
  /// of S.
  void setSema(Sema *S);
  
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
  /// @name Semantic analysis
  /// {
  bool hasSema() const { return TheSema != 0; }
  
  Sema &getSema() const { 
    assert(TheSema && "Compiler instance has no Sema object!");
    return *TheSema;
  }
  
  Sema *takeSema() { return TheSema.take(); }
  
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
  /// @name Frontend timer
  /// {

  bool hasFrontendTimer() const { return FrontendTimer != 0; }

  llvm::Timer &getFrontendTimer() const {
    assert(FrontendTimer && "Compiler instance has no frontend timer!");
    return *FrontendTimer;
  }

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

  /// clearOutputFiles - Clear the output file list, destroying the contained
  /// output streams.
  ///
  /// \param EraseFiles - If true, attempt to erase the files from disk.
  void clearOutputFiles(bool EraseFiles);

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
  /// caller is responsible for releasing the returned Diagnostic's client
  /// eventually.
  ///
  /// \param Opts - The diagnostic options; note that the created text
  /// diagnostic object contains a reference to these options and its lifetime
  /// must extend past that of the diagnostic engine.
  ///
  /// \return The new object on success, or null on failure.
  static llvm::IntrusiveRefCntPtr<Diagnostic> 
  createDiagnostics(const DiagnosticOptions &Opts, int Argc, char **Argv);

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
                                          const FrontendOptions &,
                                          SourceManager &, FileManager &);

  /// Create the AST context.
  void createASTContext();

  /// Create an external AST source to read a PCH file and attach it to the AST
  /// context.
  void createPCHExternalASTSource(llvm::StringRef Path,
                                  bool DisablePCHValidation,
                                  void *DeserializationListener);

  /// Create an external AST source to read a PCH file.
  ///
  /// \return - The new object on success, or null on failure.
  static ExternalASTSource *
  createPCHExternalASTSource(llvm::StringRef Path, const std::string &Sysroot,
                             bool DisablePCHValidation,
                             Preprocessor &PP, ASTContext &Context,
                             void *DeserializationListener);

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
                               bool ShowCodePatterns, bool ShowGlobals,
                               llvm::raw_ostream &OS);

  /// \brief Create the Sema object to be used for parsing.
  void createSema(bool CompleteTranslationUnit,
                  CodeCompleteConsumer *CompletionConsumer);
  
  /// Create the frontend timer and replace any existing one with it.
  void createFrontendTimer();

  /// Create the default output file (from the invocation's options) and add it
  /// to the list of tracked output files.
  ///
  /// \return - Null on error.
  llvm::raw_fd_ostream *
  createDefaultOutputFile(bool Binary = true, llvm::StringRef BaseInput = "",
                          llvm::StringRef Extension = "");

  /// Create a new output file and add it to the list of tracked output files,
  /// optionally deriving the output path name.
  ///
  /// \return - Null on error.
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
  /// @name Initialization Utility Methods
  /// {

  /// InitializeSourceManager - Initialize the source manager to set InputFile
  /// as the main file.
  ///
  /// \return True on success.
  bool InitializeSourceManager(llvm::StringRef InputFile);

  /// InitializeSourceManager - Initialize the source manager to set InputFile
  /// as the main file.
  ///
  /// \return True on success.
  static bool InitializeSourceManager(llvm::StringRef InputFile,
                                      Diagnostic &Diags,
                                      FileManager &FileMgr,
                                      SourceManager &SourceMgr,
                                      const FrontendOptions &Opts);

  /// }
};

} // end namespace clang

#endif
