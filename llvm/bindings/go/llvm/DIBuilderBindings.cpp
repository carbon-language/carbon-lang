//===- DIBuilderBindings.cpp - Bindings for DIBuilder ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines C bindings for the DIBuilder class.
//
//===----------------------------------------------------------------------===//

#include "DIBuilderBindings.h"

#include "IRBindings.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DIBuilder.h"

using namespace llvm;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DIBuilder, LLVMDIBuilderRef)

namespace {
template <typename T> T unwrapDI(LLVMMetadataRef v) {
  return v ? T(unwrap<MDNode>(v)) : T();
}
}

LLVMDIBuilderRef LLVMNewDIBuilder(LLVMModuleRef mref) {
  Module *m = unwrap(mref);
  return wrap(new DIBuilder(*m));
}

void LLVMDIBuilderDestroy(LLVMDIBuilderRef dref) {
  DIBuilder *d = unwrap(dref);
  delete d;
}

void LLVMDIBuilderFinalize(LLVMDIBuilderRef dref) { unwrap(dref)->finalize(); }

LLVMMetadataRef LLVMDIBuilderCreateCompileUnit(LLVMDIBuilderRef Dref,
                                               unsigned Lang, const char *File,
                                               const char *Dir,
                                               const char *Producer,
                                               int Optimized, const char *Flags,
                                               unsigned RuntimeVersion) {
  DIBuilder *D = unwrap(Dref);
  DICompileUnit CU = D->createCompileUnit(Lang, File, Dir, Producer, Optimized,
                                          Flags, RuntimeVersion);
  return wrap(CU);
}

LLVMMetadataRef LLVMDIBuilderCreateFile(LLVMDIBuilderRef Dref, const char *File,
                                        const char *Dir) {
  DIBuilder *D = unwrap(Dref);
  DIFile F = D->createFile(File, Dir);
  return wrap(F);
}

LLVMMetadataRef LLVMDIBuilderCreateLexicalBlock(LLVMDIBuilderRef Dref,
                                                LLVMMetadataRef Scope,
                                                LLVMMetadataRef File,
                                                unsigned Line,
                                                unsigned Column) {
  DIBuilder *D = unwrap(Dref);
  DILexicalBlock LB = D->createLexicalBlock(
      unwrapDI<DIDescriptor>(Scope), unwrapDI<DIFile>(File), Line, Column);
  return wrap(LB);
}

LLVMMetadataRef LLVMDIBuilderCreateLexicalBlockFile(LLVMDIBuilderRef Dref,
                                                    LLVMMetadataRef Scope,
                                                    LLVMMetadataRef File,
                                                    unsigned Discriminator) {
  DIBuilder *D = unwrap(Dref);
  DILexicalBlockFile LBF = D->createLexicalBlockFile(
      unwrapDI<DIDescriptor>(Scope), unwrapDI<DIFile>(File), Discriminator);
  return wrap(LBF);
}

LLVMMetadataRef LLVMDIBuilderCreateFunction(
    LLVMDIBuilderRef Dref, LLVMMetadataRef Scope, const char *Name,
    const char *LinkageName, LLVMMetadataRef File, unsigned Line,
    LLVMMetadataRef CompositeType, int IsLocalToUnit, int IsDefinition,
    unsigned ScopeLine, unsigned Flags, int IsOptimized, LLVMValueRef Func) {
  DIBuilder *D = unwrap(Dref);
  DISubprogram SP = D->createFunction(
      unwrapDI<DIDescriptor>(Scope), Name, LinkageName, unwrapDI<DIFile>(File),
      Line, unwrapDI<DICompositeType>(CompositeType), IsLocalToUnit,
      IsDefinition, ScopeLine, Flags, IsOptimized, unwrap<Function>(Func));
  return wrap(SP);
}

LLVMMetadataRef LLVMDIBuilderCreateLocalVariable(
    LLVMDIBuilderRef Dref, unsigned Tag, LLVMMetadataRef Scope,
    const char *Name, LLVMMetadataRef File, unsigned Line, LLVMMetadataRef Ty,
    int AlwaysPreserve, unsigned Flags, unsigned ArgNo) {
  DIBuilder *D = unwrap(Dref);
  DIVariable V = D->createLocalVariable(
      Tag, unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), Line,
      unwrapDI<DIType>(Ty), AlwaysPreserve, Flags, ArgNo);
  return wrap(V);
}

LLVMMetadataRef LLVMDIBuilderCreateBasicType(LLVMDIBuilderRef Dref,
                                             const char *Name,
                                             uint64_t SizeInBits,
                                             uint64_t AlignInBits,
                                             unsigned Encoding) {
  DIBuilder *D = unwrap(Dref);
  DIBasicType T = D->createBasicType(Name, SizeInBits, AlignInBits, Encoding);
  return wrap(T);
}

LLVMMetadataRef LLVMDIBuilderCreatePointerType(LLVMDIBuilderRef Dref,
                                               LLVMMetadataRef PointeeType,
                                               uint64_t SizeInBits,
                                               uint64_t AlignInBits,
                                               const char *Name) {
  DIBuilder *D = unwrap(Dref);
  DIDerivedType T = D->createPointerType(unwrapDI<DIType>(PointeeType),
                                         SizeInBits, AlignInBits, Name);
  return wrap(T);
}

LLVMMetadataRef
LLVMDIBuilderCreateSubroutineType(LLVMDIBuilderRef Dref, LLVMMetadataRef File,
                                  LLVMMetadataRef ParameterTypes) {
  DIBuilder *D = unwrap(Dref);
  DICompositeType CT = D->createSubroutineType(
      unwrapDI<DIFile>(File), unwrapDI<DITypeArray>(ParameterTypes));
  return wrap(CT);
}

LLVMMetadataRef LLVMDIBuilderCreateStructType(
    LLVMDIBuilderRef Dref, LLVMMetadataRef Scope, const char *Name,
    LLVMMetadataRef File, unsigned Line, uint64_t SizeInBits,
    uint64_t AlignInBits, unsigned Flags, LLVMMetadataRef DerivedFrom,
    LLVMMetadataRef ElementTypes) {
  DIBuilder *D = unwrap(Dref);
  DICompositeType CT = D->createStructType(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), Line,
      SizeInBits, AlignInBits, Flags, unwrapDI<DIType>(DerivedFrom),
      unwrapDI<DIArray>(ElementTypes));
  return wrap(CT);
}

LLVMMetadataRef
LLVMDIBuilderCreateMemberType(LLVMDIBuilderRef Dref, LLVMMetadataRef Scope,
                              const char *Name, LLVMMetadataRef File,
                              unsigned Line, uint64_t SizeInBits,
                              uint64_t AlignInBits, uint64_t OffsetInBits,
                              unsigned Flags, LLVMMetadataRef Ty) {
  DIBuilder *D = unwrap(Dref);
  DIDerivedType DT = D->createMemberType(
      unwrapDI<DIDescriptor>(Scope), Name, unwrapDI<DIFile>(File), Line,
      SizeInBits, AlignInBits, OffsetInBits, Flags, unwrapDI<DIType>(Ty));
  return wrap(DT);
}

LLVMMetadataRef LLVMDIBuilderCreateArrayType(LLVMDIBuilderRef Dref,
                                             uint64_t SizeInBits,
                                             uint64_t AlignInBits,
                                             LLVMMetadataRef ElementType,
                                             LLVMMetadataRef Subscripts) {
  DIBuilder *D = unwrap(Dref);
  DICompositeType CT =
      D->createArrayType(SizeInBits, AlignInBits, unwrapDI<DIType>(ElementType),
                         unwrapDI<DIArray>(Subscripts));
  return wrap(CT);
}

LLVMMetadataRef LLVMDIBuilderCreateTypedef(LLVMDIBuilderRef Dref,
                                           LLVMMetadataRef Ty, const char *Name,
                                           LLVMMetadataRef File, unsigned Line,
                                           LLVMMetadataRef Context) {
  DIBuilder *D = unwrap(Dref);
  DIDerivedType DT =
      D->createTypedef(unwrapDI<DIType>(Ty), Name, unwrapDI<DIFile>(File), Line,
                       unwrapDI<DIDescriptor>(Context));
  return wrap(DT);
}

LLVMMetadataRef LLVMDIBuilderGetOrCreateSubrange(LLVMDIBuilderRef Dref,
                                                 int64_t Lo, int64_t Count) {
  DIBuilder *D = unwrap(Dref);
  DISubrange S = D->getOrCreateSubrange(Lo, Count);
  return wrap(S);
}

LLVMMetadataRef LLVMDIBuilderGetOrCreateArray(LLVMDIBuilderRef Dref,
                                              LLVMMetadataRef *Data,
                                              size_t Length) {
  DIBuilder *D = unwrap(Dref);
  Metadata **DataValue = unwrap(Data);
  ArrayRef<Metadata *> Elements(DataValue, Length);
  DIArray A = D->getOrCreateArray(Elements);
  return wrap(A);
}

LLVMMetadataRef LLVMDIBuilderGetOrCreateTypeArray(LLVMDIBuilderRef Dref,
                                                  LLVMMetadataRef *Data,
                                                  size_t Length) {
  DIBuilder *D = unwrap(Dref);
  Metadata **DataValue = unwrap(Data);
  ArrayRef<Metadata *> Elements(DataValue, Length);
  DITypeArray A = D->getOrCreateTypeArray(Elements);
  return wrap(A);
}

LLVMMetadataRef LLVMDIBuilderCreateExpression(LLVMDIBuilderRef Dref,
                                              int64_t *Addr, size_t Length) {
  DIBuilder *D = unwrap(Dref);
  DIExpression Expr = D->createExpression(ArrayRef<int64_t>(Addr, Length));
  return wrap(Expr);
}

LLVMValueRef LLVMDIBuilderInsertDeclareAtEnd(LLVMDIBuilderRef Dref,
                                             LLVMValueRef Storage,
                                             LLVMMetadataRef VarInfo,
                                             LLVMMetadataRef Expr,
                                             LLVMBasicBlockRef Block) {
  DIBuilder *D = unwrap(Dref);
  Instruction *Instr =
      D->insertDeclare(unwrap(Storage), unwrapDI<DIVariable>(VarInfo),
                       unwrapDI<DIExpression>(Expr), unwrap(Block));
  return wrap(Instr);
}

LLVMValueRef LLVMDIBuilderInsertValueAtEnd(LLVMDIBuilderRef Dref,
                                           LLVMValueRef Val, uint64_t Offset,
                                           LLVMMetadataRef VarInfo,
                                           LLVMMetadataRef Expr,
                                           LLVMBasicBlockRef Block) {
  DIBuilder *D = unwrap(Dref);
  Instruction *Instr = D->insertDbgValueIntrinsic(
      unwrap(Val), Offset, unwrapDI<DIVariable>(VarInfo),
      unwrapDI<DIExpression>(Expr), unwrap(Block));
  return wrap(Instr);
}
