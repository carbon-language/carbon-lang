//===-- MICmdCommands.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmdCommands.h
//
// Overview:    CMICmdCommands instantiated.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

#pragma once

namespace MICmnCommands
{

//++ ============================================================================
// Details: MI Command are instantiated and registered automatically with the
//          Command Factory
// Gotchas: None.
// Authors: Illya Rudkin 18/02/2014.
// Changes: None.
//--
bool RegisterAll(void);
}
