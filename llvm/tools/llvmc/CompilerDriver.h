//===- CompilerDriver.h - Compiler Driver -----------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares the CompilerDriver class which implements the bulk of the
// LLVM Compiler Driver program (llvmc).
//
//===------------------------------------------------------------------------===
#ifndef LLVM_TOOLS_LLVMC_COMPILERDRIVER_H
#define LLVM_TOOLS_LLVMC_COMPILERDRIVER_H

#include <string>
#include <vector>

namespace llvm {
  /// This class provides the high level interface to the LLVM Compiler Driver.
  /// The driver's purpose is to make it easier for compiler writers and users
  /// of LLVM to utilize the compiler toolkits and LLVM toolset by learning only
  /// the interface of one program (llvmc).
  /// 
  /// @see llvmc.cpp
  /// @brief The interface to the LLVM Compiler Driver.
  class CompilerDriver {
    /// @name Types
    /// @{
    public:
      enum Phases {
        PREPROCESSING, ///< Source language combining, filtering, substitution
        TRANSLATION,   ///< Translate source -> LLVM bytecode/assembly
        OPTIMIZATION,  ///< Optimize translation result 
        LINKING,       ///< Link bytecode and native code
        ASSEMBLY,      ///< Convert program to executable
      };

      enum OptimizationLevels {
        OPT_NONE,                 ///< Zippo optimizations, nada, nil, none.
        OPT_FAST_COMPILE,         ///< Optimize to make >compile< go faster
        OPT_SIMPLE,               ///< Standard/simple optimizations
        OPT_AGGRESSIVE,           ///< Aggressive optimizations
        OPT_LINK_TIME,            ///< Aggressive + LinkTime optimizations
        OPT_AGGRESSIVE_LINK_TIME  ///< Make it go way fast!
      };

      /// This type is the input list to the CompilerDriver. It provides
      /// a vector of filename/filetype pairs. The filetype is used to look up
      /// the configuration of the actions to be taken by the driver.
      /// @brief The Input Data to the execute method
      typedef std::vector<std::pair<std::string,std::string> > InputList;

      /// This type is read from configuration files or otherwise provided to
      /// the CompilerDriver through a "ConfigDataProvider". It serves as both
      /// the template of what to do and the actual Action to be executed.
      /// @brief A structure to hold the action data for a given source
      /// language.
      struct Action {
        Action() : inputAt(0) , outputAt(0) {}
        std::string program;            ///< The program to execve
        std::vector<std::string> args;  ///< Arguments to the program
        size_t inputAt;                 ///< Argument index to insert input file
        size_t outputAt;                ///< Argument index to insert output file
      };

      struct ConfigData {
        ConfigData() : TranslatorPreprocesses(false),
          TranslatorOptimizes(false),
          TranslatorGroksDashO(false),
          PreprocessorNeeded(false) {}
        std::string langName;       ///< The name of the source language 
        bool TranslatorPreprocesses;///< Translator program will pre-process
        bool TranslatorOptimizes;   ///< Translator program will optimize too
        bool TranslatorGroksDashO;  ///< Translator understands -O arguments
        bool PreprocessorNeeded;    ///< Preprocessor is needed for translation
        Action PreProcessor;        ///< PreProcessor command line
        Action Translator;          ///< Translator command line
        Action Optimizer;           ///< Optimizer command line
        Action Assembler;           ///< Assembler command line
        Action Linker;              ///< Linker command line
      };

      /// This pure virtual interface class defines the interface between the
      /// CompilerDriver and other software that provides ConfigData objects to
      /// it. The CompilerDriver must be configured to use an object of this
      /// type so it can obtain the configuration data. 
      /// @see setConfigDataProvider
      /// @brief Configuration Data Provider interface
      class ConfigDataProvider {
      public:
        virtual ConfigData* ProvideConfigData(const std::string& filetype) = 0;
        virtual void setConfigDir(const std::string& dirName) = 0;
      };

    /// @}
    /// @name Constructors
    /// @{
    public:
      CompilerDriver(ConfigDataProvider& cdp );
      virtual ~CompilerDriver();

    /// @}
    /// @name Methods
    /// @{
    public:
      /// @brief Handle an error
      virtual void error(const std::string& errmsg);

      /// @brief Execute the actions requested for the given input list.
      virtual int execute(const InputList& list, const std::string& output);

    /// @}
    /// @name Mutators
    /// @{
    public:
      /// @brief Set the final phase at which compilation terminates
      void setFinalPhase( Phases phase ) { finalPhase = phase; }

      /// @brief Set the optimization level for the compilation
      void setOptimization( OptimizationLevels level ) { optLevel = level; }

      /// @brief Prevent the CompilerDriver from taking any actions
      void setDryRun( bool TF ) { isDryRun = TF; }

      /// @brief Cause the CompilerDriver to print to stderr all the
      /// actions it is taking.
      void setVerbose( bool TF ) { isVerbose = TF; }

      /// @brief Cause the CompilerDriver to print to stderr very verbose
      /// information that might be useful in debugging the driver's actions
      void setDebug( bool TF ) { isDebug = TF; }

      /// @brief Cause the CompilerDriver to print to stderr the 
      /// execution time of each action taken.
      void setTimeActions( bool TF ) { timeActions = TF; }

      /// @brief Indicate that native code is to be generated instead
      /// of LLVM bytecode.
      void setEmitNativeCode( bool TF ) { emitNativeCode = TF; }

      /// @brief Indicate that raw, unoptimized code is to be generated.
      void setEmitRawCode(bool TF ) { emitRawCode = TF; }

      /// @brief Set the output machine name.
      void setOutputMachine( const std::string& machineName ) {
        machine = machineName;
      }

      /// @brief Set the list of library paths to be searched for
      /// libraries.
      void addLibraryPath( const std::string& libPath ) {
        libPaths.push_back(libPath);
      }

    /// @}
    /// @name Functions
    /// @{
    private:
      Action* GetAction(ConfigData* cd, const std::string& input, 
                       const std::string& output, Phases phase );
      void DoAction(Action* a);

    /// @}
    /// @name Data
    /// @{
    private:
      ConfigDataProvider* cdp;      ///< Where we get configuration data from
      Phases finalPhase;            ///< The final phase of compilation
      OptimizationLevels optLevel;  ///< The optimization level to apply
      bool isDryRun;                ///< Prevent actions ?
      bool isVerbose;               ///< Print actions?
      bool isDebug;                 ///< Print lotsa debug info?
      bool timeActions;             ///< Time the actions executed ?
      bool emitRawCode;             ///< Emit Raw (unoptimized) code?
      bool emitNativeCode;          ///< Emit native code instead of bytecode?
      std::string machine;          ///< Target machine name
      std::vector<std::string> libPaths; ///< list of dirs to find libraries

    /// @}

  };
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
#endif
