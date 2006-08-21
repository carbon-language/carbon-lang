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
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMC_COMPILERDRIVER_H
#define LLVM_TOOLS_LLVMC_COMPILERDRIVER_H

#include <string>
#include <vector>
#include "llvm/System/Program.h"

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
      /// @brief A vector of strings, used for argument lists
      typedef std::vector<std::string> StringVector;

      /// @brief A vector of sys::Path, used for path lists
      typedef std::vector<sys::Path> PathVector;

      /// @brief A table of strings, indexed typically by Phases
      typedef std::vector<StringVector> StringTable;

      /// @brief The phases of processing that llvmc understands
      enum Phases {
        PREPROCESSING, ///< Source language combining, filtering, substitution
        TRANSLATION,   ///< Translate source -> LLVM bytecode/assembly
        OPTIMIZATION,  ///< Optimize translation result
        ASSEMBLY,      ///< Convert program to executable
        LINKING,       ///< Link bytecode and native code
        NUM_PHASES     ///< Always last!
      };

      /// @brief The levels of optimization llvmc understands
      enum OptimizationLevels {
        OPT_FAST_COMPILE,         ///< Optimize to make >compile< go faster
        OPT_SIMPLE,               ///< Standard/simple optimizations
        OPT_AGGRESSIVE,           ///< Aggressive optimizations
        OPT_LINK_TIME,            ///< Aggressive + LinkTime optimizations
        OPT_AGGRESSIVE_LINK_TIME, ///< Make it go way fast!
        OPT_NONE                  ///< No optimizations. Keep this at the end!
      };

      /// @brief Action specific flags
      enum ConfigurationFlags {
        REQUIRED_FLAG        = 0x0001, ///< Should the action always be run?
        PREPROCESSES_FLAG    = 0x0002, ///< Does this action preprocess?
        TRANSLATES_FLAG      = 0x0004, ///< Does this action translate?
        OUTPUT_IS_ASM_FLAG   = 0x0008, ///< Action produces .ll files?
        FLAGS_MASK           = 0x000F  ///< Union of all flags
      };

      /// This type is the input list to the CompilerDriver. It provides
      /// a vector of pathname/filetype pairs. The filetype is used to look up
      /// the configuration of the actions to be taken by the driver.
      /// @brief The Input Data to the execute method
      typedef std::vector<std::pair<sys::Path,std::string> > InputList;

      /// This type is read from configuration files or otherwise provided to
      /// the CompilerDriver through a "ConfigDataProvider". It serves as both
      /// the template of what to do and the actual Action to be executed.
      /// @brief A structure to hold the action data for a given source
      /// language.
      struct Action {
        Action() : flags(0) {}
        sys::Path program; ///< The program to execve
        StringVector args; ///< Arguments to the program
        unsigned flags;    ///< Action specific flags
        void set(unsigned fl ) { flags |= fl; }
        void clear(unsigned fl) { flags &= (FLAGS_MASK ^ fl); }
        bool isSet(unsigned fl) { return (flags&fl) != 0; }
      };

      struct ConfigData {
        ConfigData();
        std::string version;    ///< The version number.
        std::string langName;   ///< The name of the source language
        StringTable opts;       ///< The o10n options for each level
        StringVector libpaths;  ///< The library paths
        Action PreProcessor;    ///< PreProcessor command line
        Action Translator;      ///< Translator command line
        Action Optimizer;       ///< Optimizer command line
        Action Assembler;       ///< Assembler command line
        Action Linker;          ///< Linker command line
      };

      /// This pure virtual interface class defines the interface between the
      /// CompilerDriver and other software that provides ConfigData objects to
      /// it. The CompilerDriver must be configured to use an object of this
      /// type so it can obtain the configuration data.
      /// @see setConfigDataProvider
      /// @brief Configuration Data Provider interface
      class ConfigDataProvider {
      public:
        virtual ~ConfigDataProvider();
        virtual ConfigData* ProvideConfigData(const std::string& filetype) = 0;
        virtual void setConfigDir(const sys::Path& dirName) = 0;
      };

      /// These flags control various actions of the compiler driver. They are
      /// used by adding the needed flag values together and passing them to the
      /// compiler driver's setDriverFlags method.
      /// @see setDriverFlags
      /// @brief Driver specific flags
      enum DriverFlags {
        DRY_RUN_FLAG         = 0x0001, ///< Do everything but execute actions
        VERBOSE_FLAG         = 0x0002, ///< Print each action
        DEBUG_FLAG           = 0x0004, ///< Print debug information
        TIME_PASSES_FLAG     = 0x0008, ///< Time the passes as they execute
        TIME_ACTIONS_FLAG    = 0x0010, ///< Time the actions as they execute
        SHOW_STATS_FLAG      = 0x0020, ///< Show pass statistics
        EMIT_NATIVE_FLAG     = 0x0040, ///< Emit native code instead of bc
        EMIT_RAW_FLAG        = 0x0080, ///< Emit raw, unoptimized bytecode
        KEEP_TEMPS_FLAG      = 0x0100, ///< Don't delete temporary files
        STRIP_OUTPUT_FLAG    = 0x0200, ///< Strip symbols from linked output
        DRIVER_FLAGS_MASK    = 0x03FF  ///< Union of the above flags
      };

    /// @}
    /// @name Constructors
    /// @{
    public:
      /// @brief Static Constructor
      static CompilerDriver* Get(ConfigDataProvider& CDP);

      /// @brief Virtual destructor
      virtual ~CompilerDriver();

    /// @}
    /// @name Methods
    /// @{
    public:
      /// @brief Execute the actions requested for the given input list.
      virtual int execute(
        const InputList& list, const sys::Path& output, std::string& ErrMsg) =0;

      /// @brief Set the final phase at which compilation terminates
      virtual void setFinalPhase(Phases phase) = 0;

      /// @brief Set the optimization level for the compilation
      virtual void setOptimization(OptimizationLevels level) = 0;

      /// @brief Set the driver flags.
      virtual void setDriverFlags(unsigned flags) = 0;

      /// @brief Set the output machine name.
      virtual void setOutputMachine(const std::string& machineName) = 0;

      /// @brief Set the options for a given phase.
      virtual void setPhaseArgs(Phases phase, const StringVector& opts) = 0;

      /// @brief Set Library Paths
      virtual void setIncludePaths(const StringVector& paths) = 0;

      /// @brief Set Library Paths
      virtual void setSymbolDefines(const StringVector& paths) = 0;

      /// @brief Set Library Paths
      virtual void setLibraryPaths(const StringVector& paths) = 0;

      /// @brief Add a path to the list of library paths
      virtual void addLibraryPath( const sys::Path& libPath )  = 0;

      /// @brief Add a path to the list of paths in which to find tools
      virtual void addToolPath( const sys::Path& toolPath) = 0;

      /// @brief Set the list of -f options to be passed through
      virtual void setfPassThrough(const StringVector& fOpts) = 0;

      /// @brief Set the list of -M options to be passed through
      virtual void setMPassThrough(const StringVector& fOpts) = 0;

      /// @brief Set the list of -W options to be passed through
      virtual void setWPassThrough(const StringVector& fOpts) = 0;

      /// @brief Determine where a linkage file is located in the file system
      virtual sys::Path GetPathForLinkageItem(
        const std::string& link_item, ///< Item to be sought
        bool native = false           ///< Looking for native?
      ) = 0;

    /// @}
  };
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
#endif
