//===-- llvm/Support/DebugInfoBuilder.h - -----------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the DebugInfoBuilder class, which is
// a helper class used to construct source level debugging information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DEBUGINFOBUILDER_H
#define LLVM_SUPPORT_DEBUGINFOBUILDER_H

#include <llvm/Module.h>
#include <string>

namespace llvm {
    
class Type;
class IntegerType;
class FloatType;
class StructType;
class PointerType;
class Module;
class GlobalVariable;
class Constant;

namespace sys {
    class Path;
}

/// Helper class used to construct source-level debugging information.
///
/// The helper contains a notion of "current context", which is a
/// DWARF descriptor object representing the scope (module, class,
/// function, etc.) that currently encloses the definitions being
/// emitted.
///
/// Initially, you should call setModule() to specify the target
/// module. Descriptors which are generated will be inserted into
/// this module. This also generates the initial set of anchor
/// descriptors, if they have not already been created for the module.
///
/// Next, you should call createCompileUnitDescriptor() to create
/// the descriptor for the current compilation unit. This method
/// sets the current context to the newly created descriptor.
///
/// Once that has been done, you can then create descriptors for
/// global definitions (functions, variables, etc.). You can use
/// setContext() to modify the current context. setContext() returns
/// a reference to the previous context, allowing easy restoration
/// of the previous context.
class DebugInfoBuilder {
private:
    Module * module;
    PointerType * anyPtrType;    // Pointer to empty struct
    StructType * anchorType;
    GlobalVariable * compileUnit;
    GlobalVariable * context;
    GlobalVariable * compileUnitAnchor;
    GlobalVariable * globalVariableAnchor;
    GlobalVariable * subprogramAnchor;
    GlobalVariable * compileUnitDescriptor;

    // Create an anchor with the specified tag.
    GlobalVariable * createAnchor(unsigned anchorTag, const char * anchorName);

    // Calculate alignement for primitive types.
    unsigned getBasicAlignment(unsigned sizeInBits);

    // Calculate the size of the specified LLVM type.
    Constant * getSize(const Type * type);

    // Calculate the alignment of the specified LLVM type.
    Constant * getAlignment(const Type * type);

public:
    /// Constructor
    DebugInfoBuilder();

    /// Return the type defined by llvm.dbg.anchor.type
    StructType * getAnchorType() const { return anchorType; }
    
    /// Set the reference to the module where we will insert debugging
    /// information. Also defines the debug info types for the module and
    /// creates the initial anchors. Also changes the current context to the
    // global context for that module.
    void setModule(Module * m);
    
    /// Emit a compile unit descriptor. This should be done once for each
    /// module before any other debug descriptors are created. This also
    /// changes the current context to the global context for the compile unit.
    GlobalVariable * createCompileUnitDescriptor(
        unsigned langId,
        const sys::Path & srcPath,
        const std::string & producer);

    /// Set a new context, returning the previous context. The context is the
    /// debug descriptor representing the current scope (module, function,
    /// class, etc.)
    GlobalVariable * setContext(GlobalVariable * ctx) {
        GlobalVariable * prev = context;
        context = ctx;
        return prev;
    }
    
    /// Emit a subprogram descriptor in the current context.
    GlobalVariable * createSubProgramDescriptor(
        const std::string & name,       // Name of the subprogram
        const std::string & qualName,   // Fully-qualified name
        unsigned line,                  // Line number
        GlobalVariable * typeDesc,      // Type descriptor
        bool isInternalScoped,          // True if internal to module.
        bool isDefined);                // True if defined in this module.

    /// Create a type descriptor for a primitive type.
    GlobalVariable * createBasicTypeDescriptor(
        std::string & name,
        unsigned line,
        unsigned sizeInBits,
        unsigned alignmentInBits,
        unsigned offsetInBits,
        unsigned typeEncoding);

    /// Create a type descriptor for an integer type
    GlobalVariable * createIntegerTypeDescriptor(
        std::string & name, const IntegerType * type, bool isSigned);

    /// Create a type descriptor for an character type
    GlobalVariable * createCharacterTypeDescriptor(
        std::string & name, const IntegerType * type, bool isSigned);

    /// Create a type descriptor for an floating-point type
    GlobalVariable * createFloatTypeDescriptor(std::string & name,
        const Type * type);

    /// Create a type descriptor for a pointer type.
    GlobalVariable * createPointerTypeDescriptor(
        std::string & name,             // Name of the type
        GlobalVariable * referenceType, // Descriptor for what is pointed to
        const PointerType * type,       // LLVM type of the pointer
        unsigned line);                 // Line number of definition (0 if none)
};

}

#endif
