//===- llvm/Transforms/Utils/IntegerDivision.h ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of 32bit integer division for targets
// that don't have native support. It's largely derived from compiler-rt's
// implementation of __udivsi3, but hand-tuned for targets that prefer less
// control flow.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMS_UTILS_INTEGERDIVISION_H
#define TRANSFORMS_UTILS_INTEGERDIVISION_H

namespace llvm {

  bool expandDivision(BinaryOperator* Div);

} // End llvm namespace

#endif
