//===-- SymbolContextScope.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolContextScope_h_
#define liblldb_SymbolContextScope_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class SymbolContextScope SymbolContextScope.h "lldb/Symbol/SymbolContextScope.h"
/// @brief Inherit from this if your object can reconstruct its symbol
///        context.
///
/// Many objects that have pointers back to parent objects that own them
/// that all inherit from this pure virtual class can reconstruct their
/// symbol context without having to keep a complete SymbolContextScope
/// object in the object state. Examples of these objects include:
/// Module, CompileUnit, Function, and Block.
///
/// Other objects can contain a valid pointer to an instance of this
/// class so they can reconstruct the symbol context in which they are
/// scoped. Example objects include: Variable and Type. Such objects
/// can be scoped at a variety of levels:
///     @li module level for a built built in types.
///     @li file level for compile unit types and variables.
///     @li function or block level for types and variables defined in
///         a function body.
///
/// Objects that adhere to this protocol can reconstruct enough of a
/// symbol context to allow functions that take a symbol context to be
/// called. Lists can also be created using a SymbolContextScope* and
/// and object pairs that allow large collections of objects to be
/// passed around with minimal overhead.
//----------------------------------------------------------------------
class SymbolContextScope
{
public:
    //------------------------------------------------------------------
    /// Reconstruct the object's symbolc context into \a sc.
    ///
    /// The object should fill in as much of the SymbolContext as it
    /// can so function calls that require a symbol context can be made
    /// for the given object.
    ///
    /// @param[out] sc
    ///     A symbol context object pointer that gets filled in.
    //------------------------------------------------------------------
    virtual void
    CalculateSymbolContext (SymbolContext *sc) = 0;

    //------------------------------------------------------------------
    /// Dump the object's symbolc context to the stream \a s.
    ///
    /// The object should dump its symbol context to the stream \a s.
    /// This function is widely used in the DumpDebug and verbose output
    /// for lldb objets.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object's symbol context.
    //------------------------------------------------------------------
    virtual void
    DumpSymbolContext (Stream *s) = 0;
};

} // namespace lldb_private

#endif  // liblldb_SymbolContextScope_h_
