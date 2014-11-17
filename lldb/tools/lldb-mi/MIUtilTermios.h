//===-- MIUtilTermios.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MIUtilTermios.h
//
// Overview:    Terminal setting termios functions.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

#pragma once

namespace MIUtilTermios
{

extern void StdinTermiosReset(void);
extern void StdinTermiosSet(void);

} // MIUtilTermios
