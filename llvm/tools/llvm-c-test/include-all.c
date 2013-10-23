/*===-- include-all.c - tool for testing libLLVM and llvm-c API -----------===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file doesn't have any actual code. It just make sure that all         *|
|* the llvm-c include files are good and doesn't generate any warnings        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

// FIXME: Autogenerate this list

#include "llvm-c/Analysis.h"
#include "llvm-c/BitReader.h"
#include "llvm-c/BitWriter.h"
#include "llvm-c/Core.h"
#include "llvm-c/Disassembler.h"
#include "llvm-c/ExecutionEngine.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/LinkTimeOptimizer.h"
#include "llvm-c/Linker.h"
#include "llvm-c/Object.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "llvm-c/Transforms/IPO.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"
#include "llvm-c/Transforms/Scalar.h"
#include "llvm-c/Transforms/Vectorize.h"
#include "llvm-c/lto.h"
