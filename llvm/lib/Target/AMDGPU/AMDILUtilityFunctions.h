//===-- AMDILUtilityFunctions.h - AMDIL Utility Functions Header --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This file provides helper macros for expanding case statements.
//
//===----------------------------------------------------------------------===//
#ifndef AMDILUTILITYFUNCTIONS_H_
#define AMDILUTILITYFUNCTIONS_H_

// Macros that are used to help with switch statements for various data types
// However, these macro's do not return anything unlike the second set below.
#define ExpandCaseTo32bitIntTypes(Instr)  \
case Instr##_i32:

#define ExpandCaseTo32bitIntTruncTypes(Instr)  \
case Instr##_i32i8: \
case Instr##_i32i16: 

#define ExpandCaseToIntTypes(Instr) \
    ExpandCaseTo32bitIntTypes(Instr)

#define ExpandCaseToIntTruncTypes(Instr) \
    ExpandCaseTo32bitIntTruncTypes(Instr)

#define ExpandCaseToFloatTypes(Instr) \
    case Instr##_f32:

#define ExpandCaseTo32bitScalarTypes(Instr) \
    ExpandCaseTo32bitIntTypes(Instr) \
case Instr##_f32:

#define ExpandCaseToAllScalarTypes(Instr) \
    ExpandCaseToFloatTypes(Instr) \
ExpandCaseToIntTypes(Instr)

#define ExpandCaseToAllScalarTruncTypes(Instr) \
    ExpandCaseToFloatTruncTypes(Instr) \
ExpandCaseToIntTruncTypes(Instr)

#define ExpandCaseToAllTypes(Instr) \
ExpandCaseToAllScalarTypes(Instr)

#define ExpandCaseToAllTruncTypes(Instr) \
ExpandCaseToAllScalarTruncTypes(Instr)

// Macros that expand into  statements with return values
#define ExpandCaseTo32bitIntReturn(Instr, Return)  \
case Instr##_i32: return Return##_i32;

#define ExpandCaseToIntReturn(Instr, Return) \
    ExpandCaseTo32bitIntReturn(Instr, Return)

#define ExpandCaseToFloatReturn(Instr, Return) \
    case Instr##_f32: return Return##_f32;\

#define ExpandCaseToAllScalarReturn(Instr, Return) \
    ExpandCaseToFloatReturn(Instr, Return) \
ExpandCaseToIntReturn(Instr, Return)

// These macros expand to common groupings of RegClass ID's
#define ExpandCaseTo1CompRegID \
case AMDGPU::GPRI32RegClassID: \
case AMDGPU::GPRF32RegClassID:

#define ExpandCaseTo32BitType(Instr) \
case Instr##_i32: \
case Instr##_f32:

#endif // AMDILUTILITYFUNCTIONS_H_
