//===-- InitializeLLDB.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INITIALIZATION_INITIALIZE_LLDB_H
#define LLDB_INITIALIZATION_INITIALIZE_LLDB_H

#include "lldb/lldb-private-types.h"

namespace lldb_private
{

//------------------------------------------------------------------
/// Initializes lldb.
///
/// This function should be called prior to using any lldb
/// classes to ensure they have a chance to do any static
/// initialization that they need to do.
//------------------------------------------------------------------
void Initialize(LoadPluginCallbackType load_plugin_callback);

//------------------------------------------------------------------
/// Initializes subset of lldb for LLGS.
///
/// This function only initializes the set of components and plugins
/// necessary for lldb-platform and lldb-gdbserver, reducing the
/// impact on the statically linked binary size.
//------------------------------------------------------------------
void InitializeForLLGS(LoadPluginCallbackType load_plugin_callback);

//------------------------------------------------------------------
/// Terminates lldb
///
/// This function optionally can be called when clients are done
/// using lldb functionality to free up any static resources
/// that have been allocated during initialization or during
/// function calls. No lldb functions should be called after
/// calling this function without again calling DCInitialize()
/// again.
//------------------------------------------------------------------
void Terminate();

//------------------------------------------------------------------
/// Terminates subset of lldb initialized by InitializeForLLGS
///
/// This function optionally can be called when clients are done
/// using lldb functionality to free up any static resources
/// that have been allocated during initialization or during
/// function calls. No lldb functions should be called after
/// calling this function without again calling DCInitialize()
/// again.
//------------------------------------------------------------------
void TerminateLLGS();
}

#endif
