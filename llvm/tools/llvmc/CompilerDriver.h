//===- CompilerDriver.h - Compiler Driver ---------------------------------===//
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
      typedef unsigned OptimizationLevel;
      enum Phases {
        PREPROCESSING, ///< Source language combining, filtering, substitution
        TRANSLATION,   ///< Translate source -> LLVM bytecode/assembly
        OPTIMIZATION,  ///< Optimize translation result 
        LINKING,       ///< Link bytecode and native code
        ASSEMBLY,      ///< Convert program to executable
      };

      enum OptimizationLevels {
        OPT_NONE,
        OPT_FAST_COMPILE,
        OPT_SIMPLE,
        OPT_AGGRESSIVE,
        OPT_LINK_TIME,
        OPT_AGGRESSIVE_LINK_TIME
      };

    /// @}
    /// @name Constructors
    /// @{
    public:
      CompilerDriver();

    /// @}
    /// @name Accessors
    /// @{
    public:
      void execute(); ///< Execute the actions requested

    /// @}
    /// @name Mutators
    /// @{
    public:
      /// @brief Set the optimization level for the compilation
      void setOptimization( OptimizationLevel level );
      void setFinalPhase( Phases phase );

    /// @}
    /// @name Data
    /// @{
    public:
      Phases finalPhase;
      OptimizationLevel optLevel;

    /// @}

  };
}
