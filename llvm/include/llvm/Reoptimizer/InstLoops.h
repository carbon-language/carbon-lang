//===-- InstLoops.h - interface to insert instrumentation --------*- C++ -*--=//
//
// Instrument every back edges with counters
//===----------------------------------------------------------------------===//

#ifndef LLVM_REOPTIMIZERINSTLOOPS_H
#define LLVM_REOPTIMIZERINSTLOOPS_H

class Pass;

// createInstLoopsPass - Create a new pass to add counters on back edges
//
Pass *createInstLoopsPass();

#endif
    
