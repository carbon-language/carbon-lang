//===-- llvm/Opt/AllOpts.h - Header file to get all opt passes ---*- C++ -*--=//
//
// This file #include's all of the small optimization header files.
//
// Note that all optimizations return true if they modified the program, false
// if not.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_ALLOPTS_H
#define LLVM_OPT_ALLOPTS_H


//===----------------------------------------------------------------------===//
// Dead Code Elimination
//
#include "llvm/Optimizations/DCE.h"


//===----------------------------------------------------------------------===//
// Constant Propogation
//
#include "llvm/Optimizations/ConstantProp.h"


//===----------------------------------------------------------------------===//
// Method Inlining Pass
//
#include "llvm/Optimizations/MethodInlining.h"

//===----------------------------------------------------------------------===//
// Symbol Stripping Pass
//
#include "llvm/Optimizations/SymbolStripping.h"

//===----------------------------------------------------------------------===//
// Induction Variable Cannonicalization
//

#include "llvm/Optimizations/InductionVars.h"

//===----------------------------------------------------------------------===//
// LevelChange - Code lowering and raising
//
#include "llvm/Optimizations/LevelChange.h"

#endif
