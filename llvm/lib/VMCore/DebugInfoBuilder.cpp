//===-- llvm/VMCore/DebugInfoBuilder.cpp - ----------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the DebugInfoBuilder class, which is
// a helper class used to construct source level debugging information.
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/DebugInfoBuilder.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Constants.h>
#include <llvm/GlobalVariable.h>
#include <llvm/Module.h>
#include <llvm/Support/Dwarf.h>
#include <llvm/System/Path.h>

namespace llvm {
    
//===----------------------------------------------------------------------===//
// Debug version -- copied from MachineModuleInfo (for now), in order to avoid
// creating a dependency on CodeGen. These declarations really should be moved
// to a better place where modules can get at them without being dependent on
// CodeGen.
enum {
  LLVMDebugVersion = (6 << 16),         // Current version of debug information.
  LLVMDebugVersion5 = (5 << 16),        // Constant for version 5.
  LLVMDebugVersion4 = (4 << 16),        // Constant for version 4.
  LLVMDebugVersionMask = 0xffff0000     // Mask for version number.
};

const char ANCHOR_TYPE_NAME[] = "llvm.dbg.anchor.type";
const char COMPILE_UNIT_ANCHOR_NAME[] = "llvm.dbg.compile_units";
const char GLOBAL_VAR_ANCHOR_NAME[] = "llvm.dbg.global_variables";
const char SUBPROGRAM_ANCHOR_NAME[] = "llvm.dbg.subprograms";
const char COMPILE_UNIT_TYPE_NAME[] = "llvm.dbg.compile_unit.type";
const char COMPILE_UNIT_NAME[] = "llvm.dbg.compile_unit";
const char SUBPROGRAM_NAME[] = "llvm.dbg.subprogram";
const char BASICTYPE_NAME[] = "llvm.dbg.basictype";
const char DERIVEDTYPE_NAME[] = "llvm.dbg.derivedtype";

DebugInfoBuilder::DebugInfoBuilder() {
    anyPtrType = PointerType::getUnqual(StructType::get(NULL, NULL));
    anchorType = StructType::get(Type::Int32Ty, Type::Int32Ty, NULL);
}

GlobalVariable * DebugInfoBuilder::createAnchor(unsigned anchorTag,
    const char * anchorName) {

    std::vector<Constant *> values;
    values.push_back(ConstantInt::get(Type::Int32Ty, LLVMDebugVersion));
    values.push_back(ConstantInt::get(Type::Int32Ty, anchorTag));
    
    return new GlobalVariable(anchorType, true, GlobalValue::LinkOnceLinkage,
        ConstantStruct::get(anchorType, values), anchorName, module);
}

// Calculate the size of the specified LLVM type.
Constant * DebugInfoBuilder::getSize(const Type * type) {
    Constant * one = ConstantInt::get(Type::Int32Ty, 1);
    return ConstantExpr::getPtrToInt(
        ConstantExpr::getGetElementPtr(
            ConstantPointerNull::get(PointerType::getUnqual(type)),
            &one, 1), Type::Int32Ty);
}
    
Constant * DebugInfoBuilder::getAlignment(const Type * type) {
    // Calculates the alignment of T using "sizeof({i8, T}) - sizeof(T)"
    return ConstantExpr::getSub(
        getSize(StructType::get(Type::Int8Ty, type, NULL)),
        getSize(type));
}
    
void DebugInfoBuilder::setModule(Module * m) {
    module = m;
    module->addTypeName(ANCHOR_TYPE_NAME, anchorType);
    
    compileUnitAnchor = module->getGlobalVariable(COMPILE_UNIT_ANCHOR_NAME);
    if (compileUnitAnchor == NULL) {
        compileUnitAnchor =
            createAnchor(dwarf::DW_TAG_compile_unit, COMPILE_UNIT_ANCHOR_NAME);
    }

    globalVariableAnchor = module->getGlobalVariable(GLOBAL_VAR_ANCHOR_NAME);
    if (globalVariableAnchor == NULL) {
        globalVariableAnchor =
            createAnchor(dwarf::DW_TAG_compile_unit, GLOBAL_VAR_ANCHOR_NAME);
    }

    subprogramAnchor = module->getGlobalVariable(SUBPROGRAM_ANCHOR_NAME);
    if (subprogramAnchor == NULL) {
        subprogramAnchor =
            createAnchor(dwarf::DW_TAG_compile_unit, SUBPROGRAM_ANCHOR_NAME);
    }
    
    compileUnit = module->getGlobalVariable(COMPILE_UNIT_NAME);
    setContext(compileUnit);
}
    
GlobalVariable * DebugInfoBuilder::createCompileUnitDescriptor(unsigned langId,
    const sys::Path & srcPath, const std::string & producer) {

    if (compileUnit == NULL) {
        std::vector<Constant *> values;
        values.push_back(ConstantInt::get(
            Type::Int32Ty, LLVMDebugVersion + dwarf::DW_TAG_compile_unit));
        values.push_back(
            ConstantExpr::getBitCast(compileUnitAnchor, anyPtrType));
        values.push_back(ConstantInt::get(Type::Int32Ty, langId));
        values.push_back(ConstantArray::get(srcPath.getLast()));
        values.push_back(ConstantArray::get(srcPath.getDirname() + "/"));
        values.push_back(ConstantArray::get(producer));
    
        Constant * structVal = ConstantStruct::get(values, false);
        compileUnit = new GlobalVariable(structVal->getType(), true,
            GlobalValue::InternalLinkage, structVal, COMPILE_UNIT_NAME, module);
    }

    setContext(compileUnit);
    return compileUnit;
}

GlobalVariable * DebugInfoBuilder::createSubProgramDescriptor(
    const std::string & name,
    const std::string & qualifiedName,
    unsigned line,
    GlobalVariable * typeDesc,
    bool isInternal,
    bool isDefined) {
        
    assert(compileUnit != NULL);
    assert(subprogramAnchor != NULL);
        
    std::vector<Constant *> values;
    values.push_back(ConstantInt::get(
        Type::Int32Ty, LLVMDebugVersion + dwarf::DW_TAG_subprogram));
    values.push_back(ConstantExpr::getBitCast(subprogramAnchor, anyPtrType));
    values.push_back(ConstantExpr::getBitCast(context, anyPtrType));
    values.push_back(ConstantArray::get(name));
    values.push_back(ConstantArray::get(qualifiedName));
    values.push_back(ConstantArray::get(qualifiedName));
    values.push_back(ConstantExpr::getBitCast(compileUnit, anyPtrType));
    values.push_back(ConstantInt::get(Type::Int32Ty, line));
    values.push_back(typeDesc ?
        ConstantExpr::getBitCast(typeDesc, anyPtrType) :
        ConstantPointerNull::get(anyPtrType));
    values.push_back(ConstantInt::get(Type::Int1Ty, isInternal));
    values.push_back(ConstantInt::get(Type::Int1Ty, isDefined));
    
    Constant * structVal = ConstantStruct::get(values, false);
    return new GlobalVariable(structVal->getType(), true,
        GlobalValue::InternalLinkage, structVal, SUBPROGRAM_NAME, module);
}

GlobalVariable * DebugInfoBuilder::createBasicTypeDescriptor(
    std::string & name,
    unsigned line,
    unsigned sizeInBits,
    unsigned alignmentInBits,
    unsigned offsetInBits,
    unsigned typeEncoding) {

    std::vector<Constant *> values;
    values.push_back(ConstantInt::get(
        Type::Int32Ty, LLVMDebugVersion + dwarf::DW_TAG_base_type));
    values.push_back(ConstantExpr::getBitCast(context, anyPtrType));
    values.push_back(ConstantArray::get(name));
    values.push_back(ConstantExpr::getBitCast(compileUnit, anyPtrType));
    values.push_back(ConstantInt::get(Type::Int32Ty, line));
    values.push_back(ConstantInt::get(Type::Int32Ty, sizeInBits));
    values.push_back(ConstantInt::get(Type::Int32Ty, alignmentInBits));
    values.push_back(ConstantInt::get(Type::Int32Ty, offsetInBits));
    values.push_back(ConstantInt::get(Type::Int32Ty, typeEncoding));

    Constant * structVal = ConstantStruct::get(values, false);
    return new GlobalVariable(structVal->getType(), true,
        GlobalValue::InternalLinkage, structVal, BASICTYPE_NAME, module);
}

GlobalVariable * DebugInfoBuilder::createIntegerTypeDescriptor(
    std::string & name, const IntegerType * type, bool isSigned) {

    std::vector<Constant *> values;
    values.push_back(ConstantInt::get(
        Type::Int32Ty, LLVMDebugVersion + dwarf::DW_TAG_base_type));
    values.push_back(ConstantPointerNull::get(anyPtrType));
    values.push_back(ConstantArray::get(name));
    values.push_back(ConstantPointerNull::get(anyPtrType));
    values.push_back(ConstantInt::get(Type::Int32Ty, 0));
    values.push_back(ConstantInt::get(Type::Int32Ty, type->getBitWidth()));
    values.push_back(getAlignment(type));
    values.push_back(ConstantInt::get(Type::Int32Ty, 0));
    values.push_back(ConstantInt::get(Type::Int32Ty,
        isSigned ? dwarf::DW_ATE_signed_char : dwarf::DW_ATE_unsigned_char));

    Constant * structVal = ConstantStruct::get(values, false);
    return new GlobalVariable(structVal->getType(), true,
        GlobalValue::InternalLinkage, structVal, BASICTYPE_NAME, module);
}

GlobalVariable * DebugInfoBuilder::createCharacterTypeDescriptor(
    std::string & name, const IntegerType * type, bool isSigned) {

    std::vector<Constant *> values;
    values.push_back(ConstantInt::get(
        Type::Int32Ty, LLVMDebugVersion + dwarf::DW_TAG_base_type));
    values.push_back(ConstantPointerNull::get(anyPtrType));
    values.push_back(ConstantArray::get(name));
    values.push_back(ConstantPointerNull::get(anyPtrType));
    values.push_back(ConstantInt::get(Type::Int32Ty, 0));
    values.push_back(ConstantInt::get(Type::Int32Ty, type->getBitWidth()));
    values.push_back(getAlignment(type));
    values.push_back(ConstantInt::get(Type::Int32Ty, 0/*offsetInBits*/));
    values.push_back(ConstantInt::get(Type::Int32Ty,
        isSigned ? dwarf::DW_ATE_signed_char : dwarf::DW_ATE_unsigned_char));

    Constant * structVal = ConstantStruct::get(values, false);
    return new GlobalVariable(structVal->getType(), true,
        GlobalValue::InternalLinkage, structVal, BASICTYPE_NAME, module);
}

GlobalVariable * DebugInfoBuilder::createFloatTypeDescriptor(
    std::string & name, const Type * type) {

    std::vector<Constant *> values;
    values.push_back(ConstantInt::get(
        Type::Int32Ty, LLVMDebugVersion + dwarf::DW_TAG_base_type));
    values.push_back(ConstantPointerNull::get(anyPtrType));
    values.push_back(ConstantArray::get(name));
    values.push_back(ConstantPointerNull::get(anyPtrType));
    values.push_back(ConstantInt::get(Type::Int32Ty, 0));
    values.push_back(getSize(type));
    values.push_back(getAlignment(type));
    values.push_back(ConstantInt::get(Type::Int32Ty, 0/*offsetInBits*/));
    values.push_back(ConstantInt::get(Type::Int32Ty, dwarf::DW_ATE_float));

    Constant * structVal = ConstantStruct::get(values, false);
    return new GlobalVariable(structVal->getType(), true,
        GlobalValue::InternalLinkage, structVal, BASICTYPE_NAME, module);
}

GlobalVariable * DebugInfoBuilder::createPointerTypeDescriptor(
    std::string & name,
    GlobalVariable * referenceType,
    const PointerType * type,
    unsigned line) {

    std::vector<Constant *> values;
    values.push_back(ConstantInt::get(
        Type::Int32Ty, dwarf::DW_TAG_pointer_type + LLVMDebugVersion));
    values.push_back(
        context ? ConstantExpr::getBitCast(context, anyPtrType) : NULL);
    values.push_back(ConstantArray::get(name));
    values.push_back(
        compileUnit ? ConstantExpr::getBitCast(compileUnit, anyPtrType) : NULL);
    values.push_back(ConstantInt::get(Type::Int32Ty, line));
    values.push_back(getSize(type));
    values.push_back(getAlignment(type));
    values.push_back(ConstantInt::get(Type::Int32Ty, 0));
    values.push_back(referenceType);

    Constant * structVal = ConstantStruct::get(values, false);
    return new GlobalVariable(structVal->getType(), true,
        GlobalValue::InternalLinkage, structVal, DERIVEDTYPE_NAME, module);
}

}
