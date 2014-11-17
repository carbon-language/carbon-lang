//===-- MICmnLLDBBroadcaster.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:        MICmnLLDBBroadcaster.h
//
// Overview:    CMICmnLLDBBroadcaster interface.
//
// Environment: Compilers:  Visual C++ 12.
//                          gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//              Libraries:  See MIReadmetxt.
//
// Copyright:   None.
//--

#pragma once

// In-house headers:
#include <lldb/API/SBBroadcaster.h>
#include "MICmnBase.h"
#include "MIUtilSingletonBase.h"

//++ ============================================================================
// Details: MI derived class from LLDB SBBroardcaster API.
//
//          *** This class (files) is a place holder until we know we need it or
//          *** not
//
//          A singleton class.
// Gotchas: None.
// Authors: Illya Rudkin 28/02/2014.
// Changes: None.
//--
class CMICmnLLDBBroadcaster : public CMICmnBase, public lldb::SBBroadcaster, public MI::ISingleton<CMICmnLLDBBroadcaster>
{
    friend MI::ISingleton<CMICmnLLDBBroadcaster>;

    // Methods:
  public:
    bool Initialize(void);
    bool Shutdown(void);
    // Methods:
  private:
    /* ctor */ CMICmnLLDBBroadcaster(void);
    /* ctor */ CMICmnLLDBBroadcaster(const CMICmnLLDBBroadcaster &);
    void operator=(const CMICmnLLDBBroadcaster &);

    // Overridden:
  private:
    // From CMICmnBase
    /* dtor */ virtual ~CMICmnLLDBBroadcaster(void);
};
