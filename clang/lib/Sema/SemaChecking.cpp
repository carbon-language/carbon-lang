//===--- SemaChecking.cpp - Extra Semantic Checking -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements extra semantic analysis beyond what is enforced
//  by the C type system.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <queue>
using namespace clang;

/// getLocationOfStringLiteralByte - Return a source location that points to the
/// specified byte of the specified string literal.
///
/// Strings are amazingly complex.  They can be formed from multiple tokens and
/// can have escape sequences in them in addition to the usual trigraph and
/// escaped newline business.  This routine handles this complexity.
///
SourceLocation Sema::getLocationOfStringLiteralByte(const StringLiteral *SL,
                                                    unsigned ByteNo) const {
  assert(!SL->isWide() && "This doesn't work for wide strings yet");

  // Loop over all of the tokens in this string until we find the one that
  // contains the byte we're looking for.
  unsigned TokNo = 0;
  while (1) {
    assert(TokNo < SL->getNumConcatenated() && "Invalid byte number!");
    SourceLocation StrTokLoc = SL->getStrTokenLoc(TokNo);

    // Get the spelling of the string so that we can get the data that makes up
    // the string literal, not the identifier for the macro it is potentially
    // expanded through.
    SourceLocation StrTokSpellingLoc = SourceMgr.getSpellingLoc(StrTokLoc);

    // Re-lex the token to get its length and original spelling.
    std::pair<FileID, unsigned> LocInfo =
      SourceMgr.getDecomposedLoc(StrTokSpellingLoc);
    std::pair<const char *,const char *> Buffer =
      SourceMgr.getBufferData(LocInfo.first);
    const char *StrData = Buffer.first+LocInfo.second;

    // Create a langops struct and enable trigraphs.  This is sufficient for
    // relexing tokens.
    LangOptions LangOpts;
    LangOpts.Trigraphs = true;

    // Create a lexer starting at the beginning of this token.
    Lexer TheLexer(StrTokSpellingLoc, LangOpts, Buffer.first, StrData,
                   Buffer.second);
    Token TheTok;
    TheLexer.LexFromRawLexer(TheTok);

    // Use the StringLiteralParser to compute the length of the string in bytes.
    StringLiteralParser SLP(&TheTok, 1, PP);
    unsigned TokNumBytes = SLP.GetStringLength();

    // If the byte is in this token, return the location of the byte.
    if (ByteNo < TokNumBytes ||
        (ByteNo == TokNumBytes && TokNo == SL->getNumConcatenated())) {
      unsigned Offset =
        StringLiteralParser::getOffsetOfStringByte(TheTok, ByteNo, PP);

      // Now that we know the offset of the token in the spelling, use the
      // preprocessor to get the offset in the original source.
      return PP.AdvanceToTokenCharacter(StrTokLoc, Offset);
    }

    // Move to the next string token.
    ++TokNo;
    ByteNo -= TokNumBytes;
  }
}

/// CheckablePrintfAttr - does a function call have a "printf" attribute
/// and arguments that merit checking?
bool Sema::CheckablePrintfAttr(const FormatAttr *Format, CallExpr *TheCall) {
  if (Format->getType() == "printf") return true;
  if (Format->getType() == "printf0") {
    // printf0 allows null "format" string; if so don't check format/args
    unsigned format_idx = Format->getFormatIdx() - 1;
    // Does the index refer to the implicit object argument?
    if (isa<CXXMemberCallExpr>(TheCall)) {
      if (format_idx == 0)
        return false;
      --format_idx;
    }
    if (format_idx < TheCall->getNumArgs()) {
      Expr *Format = TheCall->getArg(format_idx)->IgnoreParenCasts();
      if (!Format->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull))
        return true;
    }
  }
  return false;
}

Action::OwningExprResult
Sema::CheckBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall) {
  OwningExprResult TheCallResult(Owned(TheCall));

  switch (BuiltinID) {
  case Builtin::BI__builtin___CFStringMakeConstantString:
    assert(TheCall->getNumArgs() == 1 &&
           "Wrong # arguments to builtin CFStringMakeConstantString");
    if (CheckObjCString(TheCall->getArg(0)))
      return ExprError();
    break;
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
    if (SemaBuiltinVAStart(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_isgreater:
  case Builtin::BI__builtin_isgreaterequal:
  case Builtin::BI__builtin_isless:
  case Builtin::BI__builtin_islessequal:
  case Builtin::BI__builtin_islessgreater:
  case Builtin::BI__builtin_isunordered:
    if (SemaBuiltinUnorderedCompare(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_isfinite:
  case Builtin::BI__builtin_isinf:
  case Builtin::BI__builtin_isinf_sign:
  case Builtin::BI__builtin_isnan:
  case Builtin::BI__builtin_isnormal:
    if (SemaBuiltinUnaryFP(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_return_address:
  case Builtin::BI__builtin_frame_address:
    if (SemaBuiltinStackAddress(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_eh_return_data_regno:
    if (SemaBuiltinEHReturnDataRegNo(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_shufflevector:
    return SemaBuiltinShuffleVector(TheCall);
    // TheCall will be freed by the smart pointer here, but that's fine, since
    // SemaBuiltinShuffleVector guts it, but then doesn't release it.
  case Builtin::BI__builtin_prefetch:
    if (SemaBuiltinPrefetch(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_object_size:
    if (SemaBuiltinObjectSize(TheCall))
      return ExprError();
    break;
  case Builtin::BI__builtin_longjmp:
    if (SemaBuiltinLongjmp(TheCall))
      return ExprError();
    break;
  case Builtin::BI__sync_fetch_and_add:
  case Builtin::BI__sync_fetch_and_sub:
  case Builtin::BI__sync_fetch_and_or:
  case Builtin::BI__sync_fetch_and_and:
  case Builtin::BI__sync_fetch_and_xor:
  case Builtin::BI__sync_fetch_and_nand:
  case Builtin::BI__sync_add_and_fetch:
  case Builtin::BI__sync_sub_and_fetch:
  case Builtin::BI__sync_and_and_fetch:
  case Builtin::BI__sync_or_and_fetch:
  case Builtin::BI__sync_xor_and_fetch:
  case Builtin::BI__sync_nand_and_fetch:
  case Builtin::BI__sync_val_compare_and_swap:
  case Builtin::BI__sync_bool_compare_and_swap:
  case Builtin::BI__sync_lock_test_and_set:
  case Builtin::BI__sync_lock_release:
    if (SemaBuiltinAtomicOverloaded(TheCall))
      return ExprError();
    break;
  }

  return move(TheCallResult);
}

/// CheckFunctionCall - Check a direct function call for various correctness
/// and safety properties not strictly enforced by the C type system.
bool Sema::CheckFunctionCall(FunctionDecl *FDecl, CallExpr *TheCall) {
  // Get the IdentifierInfo* for the called function.
  IdentifierInfo *FnInfo = FDecl->getIdentifier();

  // None of the checks below are needed for functions that don't have
  // simple names (e.g., C++ conversion functions).
  if (!FnInfo)
    return false;

  // FIXME: This mechanism should be abstracted to be less fragile and
  // more efficient. For example, just map function ids to custom
  // handlers.

  // Printf checking.
  if (const FormatAttr *Format = FDecl->getAttr<FormatAttr>()) {
    if (CheckablePrintfAttr(Format, TheCall)) {
      bool HasVAListArg = Format->getFirstArg() == 0;
      if (!HasVAListArg) {
        if (const FunctionProtoType *Proto
            = FDecl->getType()->getAs<FunctionProtoType>())
          HasVAListArg = !Proto->isVariadic();
      }
      CheckPrintfArguments(TheCall, HasVAListArg, Format->getFormatIdx() - 1,
                           HasVAListArg ? 0 : Format->getFirstArg() - 1);
    }
  }

  for (const NonNullAttr *NonNull = FDecl->getAttr<NonNullAttr>(); NonNull;
       NonNull = NonNull->getNext<NonNullAttr>())
    CheckNonNullArguments(NonNull, TheCall);

  return false;
}

bool Sema::CheckBlockCall(NamedDecl *NDecl, CallExpr *TheCall) {
  // Printf checking.
  const FormatAttr *Format = NDecl->getAttr<FormatAttr>();
  if (!Format)
    return false;

  const VarDecl *V = dyn_cast<VarDecl>(NDecl);
  if (!V)
    return false;

  QualType Ty = V->getType();
  if (!Ty->isBlockPointerType())
    return false;

  if (!CheckablePrintfAttr(Format, TheCall))
    return false;

  bool HasVAListArg = Format->getFirstArg() == 0;
  if (!HasVAListArg) {
    const FunctionType *FT =
      Ty->getAs<BlockPointerType>()->getPointeeType()->getAs<FunctionType>();
    if (const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FT))
      HasVAListArg = !Proto->isVariadic();
  }
  CheckPrintfArguments(TheCall, HasVAListArg, Format->getFormatIdx() - 1,
                       HasVAListArg ? 0 : Format->getFirstArg() - 1);

  return false;
}

/// SemaBuiltinAtomicOverloaded - We have a call to a function like
/// __sync_fetch_and_add, which is an overloaded function based on the pointer
/// type of its first argument.  The main ActOnCallExpr routines have already
/// promoted the types of arguments because all of these calls are prototyped as
/// void(...).
///
/// This function goes through and does final semantic checking for these
/// builtins,
bool Sema::SemaBuiltinAtomicOverloaded(CallExpr *TheCall) {
  DeclRefExpr *DRE =cast<DeclRefExpr>(TheCall->getCallee()->IgnoreParenCasts());
  FunctionDecl *FDecl = cast<FunctionDecl>(DRE->getDecl());

  // Ensure that we have at least one argument to do type inference from.
  if (TheCall->getNumArgs() < 1)
    return Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args)
              << 0 << TheCall->getCallee()->getSourceRange();

  // Inspect the first argument of the atomic builtin.  This should always be
  // a pointer type, whose element is an integral scalar or pointer type.
  // Because it is a pointer type, we don't have to worry about any implicit
  // casts here.
  Expr *FirstArg = TheCall->getArg(0);
  if (!FirstArg->getType()->isPointerType())
    return Diag(DRE->getLocStart(), diag::err_atomic_builtin_must_be_pointer)
             << FirstArg->getType() << FirstArg->getSourceRange();

  QualType ValType = FirstArg->getType()->getAs<PointerType>()->getPointeeType();
  if (!ValType->isIntegerType() && !ValType->isPointerType() &&
      !ValType->isBlockPointerType())
    return Diag(DRE->getLocStart(),
                diag::err_atomic_builtin_must_be_pointer_intptr)
             << FirstArg->getType() << FirstArg->getSourceRange();

  // We need to figure out which concrete builtin this maps onto.  For example,
  // __sync_fetch_and_add with a 2 byte object turns into
  // __sync_fetch_and_add_2.
#define BUILTIN_ROW(x) \
  { Builtin::BI##x##_1, Builtin::BI##x##_2, Builtin::BI##x##_4, \
    Builtin::BI##x##_8, Builtin::BI##x##_16 }

  static const unsigned BuiltinIndices[][5] = {
    BUILTIN_ROW(__sync_fetch_and_add),
    BUILTIN_ROW(__sync_fetch_and_sub),
    BUILTIN_ROW(__sync_fetch_and_or),
    BUILTIN_ROW(__sync_fetch_and_and),
    BUILTIN_ROW(__sync_fetch_and_xor),
    BUILTIN_ROW(__sync_fetch_and_nand),

    BUILTIN_ROW(__sync_add_and_fetch),
    BUILTIN_ROW(__sync_sub_and_fetch),
    BUILTIN_ROW(__sync_and_and_fetch),
    BUILTIN_ROW(__sync_or_and_fetch),
    BUILTIN_ROW(__sync_xor_and_fetch),
    BUILTIN_ROW(__sync_nand_and_fetch),

    BUILTIN_ROW(__sync_val_compare_and_swap),
    BUILTIN_ROW(__sync_bool_compare_and_swap),
    BUILTIN_ROW(__sync_lock_test_and_set),
    BUILTIN_ROW(__sync_lock_release)
  };
#undef BUILTIN_ROW

  // Determine the index of the size.
  unsigned SizeIndex;
  switch (Context.getTypeSizeInChars(ValType).getQuantity()) {
  case 1: SizeIndex = 0; break;
  case 2: SizeIndex = 1; break;
  case 4: SizeIndex = 2; break;
  case 8: SizeIndex = 3; break;
  case 16: SizeIndex = 4; break;
  default:
    return Diag(DRE->getLocStart(), diag::err_atomic_builtin_pointer_size)
             << FirstArg->getType() << FirstArg->getSourceRange();
  }

  // Each of these builtins has one pointer argument, followed by some number of
  // values (0, 1 or 2) followed by a potentially empty varags list of stuff
  // that we ignore.  Find out which row of BuiltinIndices to read from as well
  // as the number of fixed args.
  unsigned BuiltinID = FDecl->getBuiltinID();
  unsigned BuiltinIndex, NumFixed = 1;
  switch (BuiltinID) {
  default: assert(0 && "Unknown overloaded atomic builtin!");
  case Builtin::BI__sync_fetch_and_add: BuiltinIndex = 0; break;
  case Builtin::BI__sync_fetch_and_sub: BuiltinIndex = 1; break;
  case Builtin::BI__sync_fetch_and_or:  BuiltinIndex = 2; break;
  case Builtin::BI__sync_fetch_and_and: BuiltinIndex = 3; break;
  case Builtin::BI__sync_fetch_and_xor: BuiltinIndex = 4; break;
  case Builtin::BI__sync_fetch_and_nand:BuiltinIndex = 5; break;

  case Builtin::BI__sync_add_and_fetch: BuiltinIndex = 6; break;
  case Builtin::BI__sync_sub_and_fetch: BuiltinIndex = 7; break;
  case Builtin::BI__sync_and_and_fetch: BuiltinIndex = 8; break;
  case Builtin::BI__sync_or_and_fetch:  BuiltinIndex = 9; break;
  case Builtin::BI__sync_xor_and_fetch: BuiltinIndex =10; break;
  case Builtin::BI__sync_nand_and_fetch:BuiltinIndex =11; break;

  case Builtin::BI__sync_val_compare_and_swap:
    BuiltinIndex = 12;
    NumFixed = 2;
    break;
  case Builtin::BI__sync_bool_compare_and_swap:
    BuiltinIndex = 13;
    NumFixed = 2;
    break;
  case Builtin::BI__sync_lock_test_and_set: BuiltinIndex = 14; break;
  case Builtin::BI__sync_lock_release:
    BuiltinIndex = 15;
    NumFixed = 0;
    break;
  }

  // Now that we know how many fixed arguments we expect, first check that we
  // have at least that many.
  if (TheCall->getNumArgs() < 1+NumFixed)
    return Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args)
            << 0 << TheCall->getCallee()->getSourceRange();


  // Get the decl for the concrete builtin from this, we can tell what the
  // concrete integer type we should convert to is.
  unsigned NewBuiltinID = BuiltinIndices[BuiltinIndex][SizeIndex];
  const char *NewBuiltinName = Context.BuiltinInfo.GetName(NewBuiltinID);
  IdentifierInfo *NewBuiltinII = PP.getIdentifierInfo(NewBuiltinName);
  FunctionDecl *NewBuiltinDecl =
    cast<FunctionDecl>(LazilyCreateBuiltin(NewBuiltinII, NewBuiltinID,
                                           TUScope, false, DRE->getLocStart()));
  const FunctionProtoType *BuiltinFT =
    NewBuiltinDecl->getType()->getAs<FunctionProtoType>();
  ValType = BuiltinFT->getArgType(0)->getAs<PointerType>()->getPointeeType();

  // If the first type needs to be converted (e.g. void** -> int*), do it now.
  if (BuiltinFT->getArgType(0) != FirstArg->getType()) {
    ImpCastExprToType(FirstArg, BuiltinFT->getArgType(0), CastExpr::CK_BitCast);
    TheCall->setArg(0, FirstArg);
  }

  // Next, walk the valid ones promoting to the right type.
  for (unsigned i = 0; i != NumFixed; ++i) {
    Expr *Arg = TheCall->getArg(i+1);

    // If the argument is an implicit cast, then there was a promotion due to
    // "...", just remove it now.
    if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Arg)) {
      Arg = ICE->getSubExpr();
      ICE->setSubExpr(0);
      ICE->Destroy(Context);
      TheCall->setArg(i+1, Arg);
    }

    // GCC does an implicit conversion to the pointer or integer ValType.  This
    // can fail in some cases (1i -> int**), check for this error case now.
    CastExpr::CastKind Kind = CastExpr::CK_Unknown;
    CXXMethodDecl *ConversionDecl = 0;
    if (CheckCastTypes(Arg->getSourceRange(), ValType, Arg, Kind,
                       ConversionDecl))
      return true;

    // Okay, we have something that *can* be converted to the right type.  Check
    // to see if there is a potentially weird extension going on here.  This can
    // happen when you do an atomic operation on something like an char* and
    // pass in 42.  The 42 gets converted to char.  This is even more strange
    // for things like 45.123 -> char, etc.
    // FIXME: Do this check.
    ImpCastExprToType(Arg, ValType, Kind, /*isLvalue=*/false);
    TheCall->setArg(i+1, Arg);
  }

  // Switch the DeclRefExpr to refer to the new decl.
  DRE->setDecl(NewBuiltinDecl);
  DRE->setType(NewBuiltinDecl->getType());

  // Set the callee in the CallExpr.
  // FIXME: This leaks the original parens and implicit casts.
  Expr *PromotedCall = DRE;
  UsualUnaryConversions(PromotedCall);
  TheCall->setCallee(PromotedCall);


  // Change the result type of the call to match the result type of the decl.
  TheCall->setType(NewBuiltinDecl->getResultType());
  return false;
}


/// CheckObjCString - Checks that the argument to the builtin
/// CFString constructor is correct
/// FIXME: GCC currently emits the following warning:
/// "warning: input conversion stopped due to an input byte that does not
///           belong to the input codeset UTF-8"
/// Note: It might also make sense to do the UTF-16 conversion here (would
/// simplify the backend).
bool Sema::CheckObjCString(Expr *Arg) {
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Literal = dyn_cast<StringLiteral>(Arg);

  if (!Literal || Literal->isWide()) {
    Diag(Arg->getLocStart(), diag::err_cfstring_literal_not_string_constant)
      << Arg->getSourceRange();
    return true;
  }

  const char *Data = Literal->getStrData();
  unsigned Length = Literal->getByteLength();

  for (unsigned i = 0; i < Length; ++i) {
    if (!Data[i]) {
      Diag(getLocationOfStringLiteralByte(Literal, i),
           diag::warn_cfstring_literal_contains_nul_character)
        << Arg->getSourceRange();
      break;
    }
  }

  return false;
}

/// SemaBuiltinVAStart - Check the arguments to __builtin_va_start for validity.
/// Emit an error and return true on failure, return false on success.
bool Sema::SemaBuiltinVAStart(CallExpr *TheCall) {
  Expr *Fn = TheCall->getCallee();
  if (TheCall->getNumArgs() > 2) {
    Diag(TheCall->getArg(2)->getLocStart(),
         diag::err_typecheck_call_too_many_args)
      << 0 /*function call*/ << Fn->getSourceRange()
      << SourceRange(TheCall->getArg(2)->getLocStart(),
                     (*(TheCall->arg_end()-1))->getLocEnd());
    return true;
  }

  if (TheCall->getNumArgs() < 2) {
    return Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args)
      << 0 /*function call*/;
  }

  // Determine whether the current function is variadic or not.
  bool isVariadic;
  if (CurBlock)
    isVariadic = CurBlock->isVariadic;
  else if (getCurFunctionDecl()) {
    if (FunctionProtoType* FTP =
            dyn_cast<FunctionProtoType>(getCurFunctionDecl()->getType()))
      isVariadic = FTP->isVariadic();
    else
      isVariadic = false;
  } else {
    isVariadic = getCurMethodDecl()->isVariadic();
  }

  if (!isVariadic) {
    Diag(Fn->getLocStart(), diag::err_va_start_used_in_non_variadic_function);
    return true;
  }

  // Verify that the second argument to the builtin is the last argument of the
  // current function or method.
  bool SecondArgIsLastNamedArgument = false;
  const Expr *Arg = TheCall->getArg(1)->IgnoreParenCasts();

  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Arg)) {
    if (const ParmVarDecl *PV = dyn_cast<ParmVarDecl>(DR->getDecl())) {
      // FIXME: This isn't correct for methods (results in bogus warning).
      // Get the last formal in the current function.
      const ParmVarDecl *LastArg;
      if (CurBlock)
        LastArg = *(CurBlock->TheDecl->param_end()-1);
      else if (FunctionDecl *FD = getCurFunctionDecl())
        LastArg = *(FD->param_end()-1);
      else
        LastArg = *(getCurMethodDecl()->param_end()-1);
      SecondArgIsLastNamedArgument = PV == LastArg;
    }
  }

  if (!SecondArgIsLastNamedArgument)
    Diag(TheCall->getArg(1)->getLocStart(),
         diag::warn_second_parameter_of_va_start_not_last_named_argument);
  return false;
}

/// SemaBuiltinUnorderedCompare - Handle functions like __builtin_isgreater and
/// friends.  This is declared to take (...), so we have to check everything.
bool Sema::SemaBuiltinUnorderedCompare(CallExpr *TheCall) {
  if (TheCall->getNumArgs() < 2)
    return Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args)
      << 0 /*function call*/;
  if (TheCall->getNumArgs() > 2)
    return Diag(TheCall->getArg(2)->getLocStart(),
                diag::err_typecheck_call_too_many_args)
      << 0 /*function call*/
      << SourceRange(TheCall->getArg(2)->getLocStart(),
                     (*(TheCall->arg_end()-1))->getLocEnd());

  Expr *OrigArg0 = TheCall->getArg(0);
  Expr *OrigArg1 = TheCall->getArg(1);

  // Do standard promotions between the two arguments, returning their common
  // type.
  QualType Res = UsualArithmeticConversions(OrigArg0, OrigArg1, false);

  // Make sure any conversions are pushed back into the call; this is
  // type safe since unordered compare builtins are declared as "_Bool
  // foo(...)".
  TheCall->setArg(0, OrigArg0);
  TheCall->setArg(1, OrigArg1);

  if (OrigArg0->isTypeDependent() || OrigArg1->isTypeDependent())
    return false;

  // If the common type isn't a real floating type, then the arguments were
  // invalid for this operation.
  if (!Res->isRealFloatingType())
    return Diag(OrigArg0->getLocStart(),
                diag::err_typecheck_call_invalid_ordered_compare)
      << OrigArg0->getType() << OrigArg1->getType()
      << SourceRange(OrigArg0->getLocStart(), OrigArg1->getLocEnd());

  return false;
}

/// SemaBuiltinUnorderedCompare - Handle functions like __builtin_isnan and
/// friends.  This is declared to take (...), so we have to check everything.
bool Sema::SemaBuiltinUnaryFP(CallExpr *TheCall) {
  if (TheCall->getNumArgs() < 1)
    return Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_few_args)
      << 0 /*function call*/;
  if (TheCall->getNumArgs() > 1)
    return Diag(TheCall->getArg(1)->getLocStart(),
                diag::err_typecheck_call_too_many_args)
      << 0 /*function call*/
      << SourceRange(TheCall->getArg(1)->getLocStart(),
                     (*(TheCall->arg_end()-1))->getLocEnd());

  Expr *OrigArg = TheCall->getArg(0);

  if (OrigArg->isTypeDependent())
    return false;

  // This operation requires a floating-point number
  if (!OrigArg->getType()->isRealFloatingType())
    return Diag(OrigArg->getLocStart(),
                diag::err_typecheck_call_invalid_unary_fp)
      << OrigArg->getType() << OrigArg->getSourceRange();

  return false;
}

bool Sema::SemaBuiltinStackAddress(CallExpr *TheCall) {
  // The signature for these builtins is exact; the only thing we need
  // to check is that the argument is a constant.
  SourceLocation Loc;
  if (!TheCall->getArg(0)->isTypeDependent() &&
      !TheCall->getArg(0)->isValueDependent() &&
      !TheCall->getArg(0)->isIntegerConstantExpr(Context, &Loc))
    return Diag(Loc, diag::err_stack_const_level) << TheCall->getSourceRange();

  return false;
}

/// SemaBuiltinShuffleVector - Handle __builtin_shufflevector.
// This is declared to take (...), so we have to check everything.
Action::OwningExprResult Sema::SemaBuiltinShuffleVector(CallExpr *TheCall) {
  if (TheCall->getNumArgs() < 3)
    return ExprError(Diag(TheCall->getLocEnd(),
                          diag::err_typecheck_call_too_few_args)
      << 0 /*function call*/ << TheCall->getSourceRange());

  unsigned numElements = std::numeric_limits<unsigned>::max();
  if (!TheCall->getArg(0)->isTypeDependent() &&
      !TheCall->getArg(1)->isTypeDependent()) {
    QualType FAType = TheCall->getArg(0)->getType();
    QualType SAType = TheCall->getArg(1)->getType();

    if (!FAType->isVectorType() || !SAType->isVectorType()) {
      Diag(TheCall->getLocStart(), diag::err_shufflevector_non_vector)
        << SourceRange(TheCall->getArg(0)->getLocStart(),
                       TheCall->getArg(1)->getLocEnd());
      return ExprError();
    }

    if (!Context.hasSameUnqualifiedType(FAType, SAType)) {
      Diag(TheCall->getLocStart(), diag::err_shufflevector_incompatible_vector)
        << SourceRange(TheCall->getArg(0)->getLocStart(),
                       TheCall->getArg(1)->getLocEnd());
      return ExprError();
    }

    numElements = FAType->getAs<VectorType>()->getNumElements();
    if (TheCall->getNumArgs() != numElements+2) {
      if (TheCall->getNumArgs() < numElements+2)
        return ExprError(Diag(TheCall->getLocEnd(),
                              diag::err_typecheck_call_too_few_args)
                 << 0 /*function call*/ << TheCall->getSourceRange());
      return ExprError(Diag(TheCall->getLocEnd(),
                            diag::err_typecheck_call_too_many_args)
                 << 0 /*function call*/ << TheCall->getSourceRange());
    }
  }

  for (unsigned i = 2; i < TheCall->getNumArgs(); i++) {
    if (TheCall->getArg(i)->isTypeDependent() ||
        TheCall->getArg(i)->isValueDependent())
      continue;

    llvm::APSInt Result(32);
    if (!TheCall->getArg(i)->isIntegerConstantExpr(Result, Context))
      return ExprError(Diag(TheCall->getLocStart(),
                  diag::err_shufflevector_nonconstant_argument)
                << TheCall->getArg(i)->getSourceRange());

    if (Result.getActiveBits() > 64 || Result.getZExtValue() >= numElements*2)
      return ExprError(Diag(TheCall->getLocStart(),
                  diag::err_shufflevector_argument_too_large)
               << TheCall->getArg(i)->getSourceRange());
  }

  llvm::SmallVector<Expr*, 32> exprs;

  for (unsigned i = 0, e = TheCall->getNumArgs(); i != e; i++) {
    exprs.push_back(TheCall->getArg(i));
    TheCall->setArg(i, 0);
  }

  return Owned(new (Context) ShuffleVectorExpr(Context, exprs.begin(),
                                            exprs.size(), exprs[0]->getType(),
                                            TheCall->getCallee()->getLocStart(),
                                            TheCall->getRParenLoc()));
}

/// SemaBuiltinPrefetch - Handle __builtin_prefetch.
// This is declared to take (const void*, ...) and can take two
// optional constant int args.
bool Sema::SemaBuiltinPrefetch(CallExpr *TheCall) {
  unsigned NumArgs = TheCall->getNumArgs();

  if (NumArgs > 3)
    return Diag(TheCall->getLocEnd(), diag::err_typecheck_call_too_many_args)
             << 0 /*function call*/ << TheCall->getSourceRange();

  // Argument 0 is checked for us and the remaining arguments must be
  // constant integers.
  for (unsigned i = 1; i != NumArgs; ++i) {
    Expr *Arg = TheCall->getArg(i);
    if (Arg->isTypeDependent())
      continue;

    if (!Arg->getType()->isIntegralType())
      return Diag(TheCall->getLocStart(), diag::err_prefetch_invalid_arg_type)
              << Arg->getSourceRange();

    ImpCastExprToType(Arg, Context.IntTy, CastExpr::CK_IntegralCast);
    TheCall->setArg(i, Arg);

    if (Arg->isValueDependent())
      continue;

    llvm::APSInt Result;
    if (!Arg->isIntegerConstantExpr(Result, Context))
      return Diag(TheCall->getLocStart(), diag::err_prefetch_invalid_arg_ice)
        << SourceRange(Arg->getLocStart(), Arg->getLocEnd());

    // FIXME: gcc issues a warning and rewrites these to 0. These
    // seems especially odd for the third argument since the default
    // is 3.
    if (i == 1) {
      if (Result.getLimitedValue() > 1)
        return Diag(TheCall->getLocStart(), diag::err_argument_invalid_range)
             << "0" << "1" << Arg->getSourceRange();
    } else {
      if (Result.getLimitedValue() > 3)
        return Diag(TheCall->getLocStart(), diag::err_argument_invalid_range)
            << "0" << "3" << Arg->getSourceRange();
    }
  }

  return false;
}

/// SemaBuiltinEHReturnDataRegNo - Handle __builtin_eh_return_data_regno, the
/// operand must be an integer constant.
bool Sema::SemaBuiltinEHReturnDataRegNo(CallExpr *TheCall) {
  llvm::APSInt Result;
  if (!TheCall->getArg(0)->isIntegerConstantExpr(Result, Context))
    return Diag(TheCall->getLocStart(), diag::err_expr_not_ice)
      << TheCall->getArg(0)->getSourceRange();
  
  return false;
}


/// SemaBuiltinObjectSize - Handle __builtin_object_size(void *ptr,
/// int type). This simply type checks that type is one of the defined
/// constants (0-3).
// For compatability check 0-3, llvm only handles 0 and 2.
bool Sema::SemaBuiltinObjectSize(CallExpr *TheCall) {
  Expr *Arg = TheCall->getArg(1);
  if (Arg->isTypeDependent())
    return false;

  QualType ArgType = Arg->getType();
  const BuiltinType *BT = ArgType->getAs<BuiltinType>();
  llvm::APSInt Result(32);
  if (!BT || BT->getKind() != BuiltinType::Int)
    return Diag(TheCall->getLocStart(), diag::err_object_size_invalid_argument)
             << SourceRange(Arg->getLocStart(), Arg->getLocEnd());

  if (Arg->isValueDependent())
    return false;

  if (!Arg->isIntegerConstantExpr(Result, Context)) {
    return Diag(TheCall->getLocStart(), diag::err_object_size_invalid_argument)
             << SourceRange(Arg->getLocStart(), Arg->getLocEnd());
  }

  if (Result.getSExtValue() < 0 || Result.getSExtValue() > 3) {
    return Diag(TheCall->getLocStart(), diag::err_argument_invalid_range)
             << "0" << "3" << SourceRange(Arg->getLocStart(), Arg->getLocEnd());
  }

  return false;
}

/// SemaBuiltinLongjmp - Handle __builtin_longjmp(void *env[5], int val).
/// This checks that val is a constant 1.
bool Sema::SemaBuiltinLongjmp(CallExpr *TheCall) {
  Expr *Arg = TheCall->getArg(1);
  if (Arg->isTypeDependent() || Arg->isValueDependent())
    return false;

  llvm::APSInt Result(32);
  if (!Arg->isIntegerConstantExpr(Result, Context) || Result != 1)
    return Diag(TheCall->getLocStart(), diag::err_builtin_longjmp_invalid_val)
             << SourceRange(Arg->getLocStart(), Arg->getLocEnd());

  return false;
}

// Handle i > 1 ? "x" : "y", recursivelly
bool Sema::SemaCheckStringLiteral(const Expr *E, const CallExpr *TheCall,
                                  bool HasVAListArg,
                                  unsigned format_idx, unsigned firstDataArg) {
  if (E->isTypeDependent() || E->isValueDependent())
    return false;

  switch (E->getStmtClass()) {
  case Stmt::ConditionalOperatorClass: {
    const ConditionalOperator *C = cast<ConditionalOperator>(E);
    return SemaCheckStringLiteral(C->getTrueExpr(), TheCall,
                                  HasVAListArg, format_idx, firstDataArg)
        && SemaCheckStringLiteral(C->getRHS(), TheCall,
                                  HasVAListArg, format_idx, firstDataArg);
  }

  case Stmt::ImplicitCastExprClass: {
    const ImplicitCastExpr *Expr = cast<ImplicitCastExpr>(E);
    return SemaCheckStringLiteral(Expr->getSubExpr(), TheCall, HasVAListArg,
                                  format_idx, firstDataArg);
  }

  case Stmt::ParenExprClass: {
    const ParenExpr *Expr = cast<ParenExpr>(E);
    return SemaCheckStringLiteral(Expr->getSubExpr(), TheCall, HasVAListArg,
                                  format_idx, firstDataArg);
  }

  case Stmt::DeclRefExprClass: {
    const DeclRefExpr *DR = cast<DeclRefExpr>(E);

    // As an exception, do not flag errors for variables binding to
    // const string literals.
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      bool isConstant = false;
      QualType T = DR->getType();

      if (const ArrayType *AT = Context.getAsArrayType(T)) {
        isConstant = AT->getElementType().isConstant(Context);
      } else if (const PointerType *PT = T->getAs<PointerType>()) {
        isConstant = T.isConstant(Context) &&
                     PT->getPointeeType().isConstant(Context);
      }

      if (isConstant) {
        const VarDecl *Def = 0;
        if (const Expr *Init = VD->getDefinition(Def))
          return SemaCheckStringLiteral(Init, TheCall,
                                        HasVAListArg, format_idx, firstDataArg);
      }

      // For vprintf* functions (i.e., HasVAListArg==true), we add a
      // special check to see if the format string is a function parameter
      // of the function calling the printf function.  If the function
      // has an attribute indicating it is a printf-like function, then we
      // should suppress warnings concerning non-literals being used in a call
      // to a vprintf function.  For example:
      //
      // void
      // logmessage(char const *fmt __attribute__ (format (printf, 1, 2)), ...){
      //      va_list ap;
      //      va_start(ap, fmt);
      //      vprintf(fmt, ap);  // Do NOT emit a warning about "fmt".
      //      ...
      //
      //
      //  FIXME: We don't have full attribute support yet, so just check to see
      //    if the argument is a DeclRefExpr that references a parameter.  We'll
      //    add proper support for checking the attribute later.
      if (HasVAListArg)
        if (isa<ParmVarDecl>(VD))
          return true;
    }

    return false;
  }

  case Stmt::CallExprClass: {
    const CallExpr *CE = cast<CallExpr>(E);
    if (const ImplicitCastExpr *ICE
          = dyn_cast<ImplicitCastExpr>(CE->getCallee())) {
      if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr())) {
        if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
          if (const FormatArgAttr *FA = FD->getAttr<FormatArgAttr>()) {
            unsigned ArgIndex = FA->getFormatIdx();
            const Expr *Arg = CE->getArg(ArgIndex - 1);

            return SemaCheckStringLiteral(Arg, TheCall, HasVAListArg,
                                          format_idx, firstDataArg);
          }
        }
      }
    }

    return false;
  }
  case Stmt::ObjCStringLiteralClass:
  case Stmt::StringLiteralClass: {
    const StringLiteral *StrE = NULL;

    if (const ObjCStringLiteral *ObjCFExpr = dyn_cast<ObjCStringLiteral>(E))
      StrE = ObjCFExpr->getString();
    else
      StrE = cast<StringLiteral>(E);

    if (StrE) {
      CheckPrintfString(StrE, E, TheCall, HasVAListArg, format_idx,
                        firstDataArg);
      return true;
    }

    return false;
  }

  default:
    return false;
  }
}

void
Sema::CheckNonNullArguments(const NonNullAttr *NonNull,
                            const CallExpr *TheCall) {
  for (NonNullAttr::iterator i = NonNull->begin(), e = NonNull->end();
       i != e; ++i) {
    const Expr *ArgExpr = TheCall->getArg(*i);
    if (ArgExpr->isNullPointerConstant(Context, 
                                       Expr::NPC_ValueDependentIsNotNull))
      Diag(TheCall->getCallee()->getLocStart(), diag::warn_null_arg)
        << ArgExpr->getSourceRange();
  }
}

/// CheckPrintfArguments - Check calls to printf (and similar functions) for
/// correct use of format strings.
///
///  HasVAListArg - A predicate indicating whether the printf-like
///    function is passed an explicit va_arg argument (e.g., vprintf)
///
///  format_idx - The index into Args for the format string.
///
/// Improper format strings to functions in the printf family can be
/// the source of bizarre bugs and very serious security holes.  A
/// good source of information is available in the following paper
/// (which includes additional references):
///
///  FormatGuard: Automatic Protection From printf Format String
///  Vulnerabilities, Proceedings of the 10th USENIX Security Symposium, 2001.
///
/// Functionality implemented:
///
///  We can statically check the following properties for string
///  literal format strings for non v.*printf functions (where the
///  arguments are passed directly):
//
///  (1) Are the number of format conversions equal to the number of
///      data arguments?
///
///  (2) Does each format conversion correctly match the type of the
///      corresponding data argument?  (TODO)
///
/// Moreover, for all printf functions we can:
///
///  (3) Check for a missing format string (when not caught by type checking).
///
///  (4) Check for no-operation flags; e.g. using "#" with format
///      conversion 'c'  (TODO)
///
///  (5) Check the use of '%n', a major source of security holes.
///
///  (6) Check for malformed format conversions that don't specify anything.
///
///  (7) Check for empty format strings.  e.g: printf("");
///
///  (8) Check that the format string is a wide literal.
///
/// All of these checks can be done by parsing the format string.
///
/// For now, we ONLY do (1), (3), (5), (6), (7), and (8).
void
Sema::CheckPrintfArguments(const CallExpr *TheCall, bool HasVAListArg,
                           unsigned format_idx, unsigned firstDataArg) {
  const Expr *Fn = TheCall->getCallee();

  // The way the format attribute works in GCC, the implicit this argument
  // of member functions is counted. However, it doesn't appear in our own
  // lists, so decrement format_idx in that case.
  if (isa<CXXMemberCallExpr>(TheCall)) {
    // Catch a format attribute mistakenly referring to the object argument.
    if (format_idx == 0)
      return;
    --format_idx;
    if(firstDataArg != 0)
      --firstDataArg;
  }

  // CHECK: printf-like function is called with no format string.
  if (format_idx >= TheCall->getNumArgs()) {
    Diag(TheCall->getRParenLoc(), diag::warn_printf_missing_format_string)
      << Fn->getSourceRange();
    return;
  }

  const Expr *OrigFormatExpr = TheCall->getArg(format_idx)->IgnoreParenCasts();

  // CHECK: format string is not a string literal.
  //
  // Dynamically generated format strings are difficult to
  // automatically vet at compile time.  Requiring that format strings
  // are string literals: (1) permits the checking of format strings by
  // the compiler and thereby (2) can practically remove the source of
  // many format string exploits.

  // Format string can be either ObjC string (e.g. @"%d") or
  // C string (e.g. "%d")
  // ObjC string uses the same format specifiers as C string, so we can use
  // the same format string checking logic for both ObjC and C strings.
  if (SemaCheckStringLiteral(OrigFormatExpr, TheCall, HasVAListArg, format_idx,
                             firstDataArg))
    return;  // Literal format string found, check done!

  // If there are no arguments specified, warn with -Wformat-security, otherwise
  // warn only with -Wformat-nonliteral.
  if (TheCall->getNumArgs() == format_idx+1)
    Diag(TheCall->getArg(format_idx)->getLocStart(),
         diag::warn_printf_nonliteral_noargs)
      << OrigFormatExpr->getSourceRange();
  else
    Diag(TheCall->getArg(format_idx)->getLocStart(),
         diag::warn_printf_nonliteral)
           << OrigFormatExpr->getSourceRange();
}

void Sema::CheckPrintfString(const StringLiteral *FExpr,
                             const Expr *OrigFormatExpr,
                             const CallExpr *TheCall, bool HasVAListArg,
                             unsigned format_idx, unsigned firstDataArg) {

  const ObjCStringLiteral *ObjCFExpr =
    dyn_cast<ObjCStringLiteral>(OrigFormatExpr);

  // CHECK: is the format string a wide literal?
  if (FExpr->isWide()) {
    Diag(FExpr->getLocStart(),
         diag::warn_printf_format_string_is_wide_literal)
      << OrigFormatExpr->getSourceRange();
    return;
  }

  // Str - The format string.  NOTE: this is NOT null-terminated!
  const char *Str = FExpr->getStrData();

  // CHECK: empty format string?
  unsigned StrLen = FExpr->getByteLength();

  if (StrLen == 0) {
    Diag(FExpr->getLocStart(), diag::warn_printf_empty_format_string)
      << OrigFormatExpr->getSourceRange();
    return;
  }

  // We process the format string using a binary state machine.  The
  // current state is stored in CurrentState.
  enum {
    state_OrdChr,
    state_Conversion
  } CurrentState = state_OrdChr;

  // numConversions - The number of conversions seen so far.  This is
  //  incremented as we traverse the format string.
  unsigned numConversions = 0;

  // numDataArgs - The number of data arguments after the format
  //  string.  This can only be determined for non vprintf-like
  //  functions.  For those functions, this value is 1 (the sole
  //  va_arg argument).
  unsigned numDataArgs = TheCall->getNumArgs()-firstDataArg;

  // Inspect the format string.
  unsigned StrIdx = 0;

  // LastConversionIdx - Index within the format string where we last saw
  //  a '%' character that starts a new format conversion.
  unsigned LastConversionIdx = 0;

  for (; StrIdx < StrLen; ++StrIdx) {

    // Is the number of detected conversion conversions greater than
    // the number of matching data arguments?  If so, stop.
    if (!HasVAListArg && numConversions > numDataArgs) break;

    // Handle "\0"
    if (Str[StrIdx] == '\0') {
      // The string returned by getStrData() is not null-terminated,
      // so the presence of a null character is likely an error.
      Diag(getLocationOfStringLiteralByte(FExpr, StrIdx),
           diag::warn_printf_format_string_contains_null_char)
        <<  OrigFormatExpr->getSourceRange();
      return;
    }

    // Ordinary characters (not processing a format conversion).
    if (CurrentState == state_OrdChr) {
      if (Str[StrIdx] == '%') {
        CurrentState = state_Conversion;
        LastConversionIdx = StrIdx;
      }
      continue;
    }

    // Seen '%'.  Now processing a format conversion.
    switch (Str[StrIdx]) {
    // Handle dynamic precision or width specifier.
    case '*': {
      ++numConversions;

      if (!HasVAListArg) {
        if (numConversions > numDataArgs) {
          SourceLocation Loc = getLocationOfStringLiteralByte(FExpr, StrIdx);

          if (Str[StrIdx-1] == '.')
            Diag(Loc, diag::warn_printf_asterisk_precision_missing_arg)
              << OrigFormatExpr->getSourceRange();
          else
            Diag(Loc, diag::warn_printf_asterisk_width_missing_arg)
              << OrigFormatExpr->getSourceRange();

          // Don't do any more checking.  We'll just emit spurious errors.
          return;
        }

        // Perform type checking on width/precision specifier.
        const Expr *E = TheCall->getArg(format_idx+numConversions);
        if (const BuiltinType *BT = E->getType()->getAs<BuiltinType>())
          if (BT->getKind() == BuiltinType::Int)
            break;

        SourceLocation Loc = getLocationOfStringLiteralByte(FExpr, StrIdx);

        if (Str[StrIdx-1] == '.')
          Diag(Loc, diag::warn_printf_asterisk_precision_wrong_type)
          << E->getType() << E->getSourceRange();
        else
          Diag(Loc, diag::warn_printf_asterisk_width_wrong_type)
          << E->getType() << E->getSourceRange();

        break;
      }
    }

    // Characters which can terminate a format conversion
    // (e.g. "%d").  Characters that specify length modifiers or
    // other flags are handled by the default case below.
    //
    // FIXME: additional checks will go into the following cases.
    case 'i':
    case 'd':
    case 'o':
    case 'u':
    case 'x':
    case 'X':
    case 'e':
    case 'E':
    case 'f':
    case 'F':
    case 'g':
    case 'G':
    case 'a':
    case 'A':
    case 'c':
    case 's':
    case 'p':
      ++numConversions;
      CurrentState = state_OrdChr;
      break;

    case 'm':
      // FIXME: Warn in situations where this isn't supported!
      CurrentState = state_OrdChr;
      break;

    // CHECK: Are we using "%n"?  Issue a warning.
    case 'n': {
      ++numConversions;
      CurrentState = state_OrdChr;
      SourceLocation Loc = getLocationOfStringLiteralByte(FExpr,
                                                          LastConversionIdx);

      Diag(Loc, diag::warn_printf_write_back)<<OrigFormatExpr->getSourceRange();
      break;
    }

    // Handle "%@"
    case '@':
      // %@ is allowed in ObjC format strings only.
      if (ObjCFExpr != NULL)
        CurrentState = state_OrdChr;
      else {
        // Issue a warning: invalid format conversion.
        SourceLocation Loc =
          getLocationOfStringLiteralByte(FExpr, LastConversionIdx);

        Diag(Loc, diag::warn_printf_invalid_conversion)
          <<  std::string(Str+LastConversionIdx,
                          Str+std::min(LastConversionIdx+2, StrLen))
          << OrigFormatExpr->getSourceRange();
      }
      ++numConversions;
      break;

    // Handle "%%"
    case '%':
      // Sanity check: Was the first "%" character the previous one?
      // If not, we will assume that we have a malformed format
      // conversion, and that the current "%" character is the start
      // of a new conversion.
      if (StrIdx - LastConversionIdx == 1)
        CurrentState = state_OrdChr;
      else {
        // Issue a warning: invalid format conversion.
        SourceLocation Loc =
          getLocationOfStringLiteralByte(FExpr, LastConversionIdx);

        Diag(Loc, diag::warn_printf_invalid_conversion)
          << std::string(Str+LastConversionIdx, Str+StrIdx)
          << OrigFormatExpr->getSourceRange();

        // This conversion is broken.  Advance to the next format
        // conversion.
        LastConversionIdx = StrIdx;
        ++numConversions;
      }
      break;

    default:
      // This case catches all other characters: flags, widths, etc.
      // We should eventually process those as well.
      break;
    }
  }

  if (CurrentState == state_Conversion) {
    // Issue a warning: invalid format conversion.
    SourceLocation Loc =
      getLocationOfStringLiteralByte(FExpr, LastConversionIdx);

    Diag(Loc, diag::warn_printf_invalid_conversion)
      << std::string(Str+LastConversionIdx,
                     Str+std::min(LastConversionIdx+2, StrLen))
      << OrigFormatExpr->getSourceRange();
    return;
  }

  if (!HasVAListArg) {
    // CHECK: Does the number of format conversions exceed the number
    //        of data arguments?
    if (numConversions > numDataArgs) {
      SourceLocation Loc =
        getLocationOfStringLiteralByte(FExpr, LastConversionIdx);

      Diag(Loc, diag::warn_printf_insufficient_data_args)
        << OrigFormatExpr->getSourceRange();
    }
    // CHECK: Does the number of data arguments exceed the number of
    //        format conversions in the format string?
    else if (numConversions < numDataArgs)
      Diag(TheCall->getArg(format_idx+numConversions+1)->getLocStart(),
           diag::warn_printf_too_many_data_args)
        << OrigFormatExpr->getSourceRange();
  }
}

//===--- CHECK: Return Address of Stack Variable --------------------------===//

static DeclRefExpr* EvalVal(Expr *E);
static DeclRefExpr* EvalAddr(Expr* E);

/// CheckReturnStackAddr - Check if a return statement returns the address
///   of a stack variable.
void
Sema::CheckReturnStackAddr(Expr *RetValExp, QualType lhsType,
                           SourceLocation ReturnLoc) {

  // Perform checking for returned stack addresses.
  if (lhsType->isPointerType() || lhsType->isBlockPointerType()) {
    if (DeclRefExpr *DR = EvalAddr(RetValExp))
      Diag(DR->getLocStart(), diag::warn_ret_stack_addr)
       << DR->getDecl()->getDeclName() << RetValExp->getSourceRange();

    // Skip over implicit cast expressions when checking for block expressions.
    RetValExp = RetValExp->IgnoreParenCasts();

    if (BlockExpr *C = dyn_cast<BlockExpr>(RetValExp))
      if (C->hasBlockDeclRefExprs())
        Diag(C->getLocStart(), diag::err_ret_local_block)
          << C->getSourceRange();
    
    if (AddrLabelExpr *ALE = dyn_cast<AddrLabelExpr>(RetValExp))
      Diag(ALE->getLocStart(), diag::warn_ret_addr_label)
        << ALE->getSourceRange();
    
  } else if (lhsType->isReferenceType()) {
    // Perform checking for stack values returned by reference.
    // Check for a reference to the stack
    if (DeclRefExpr *DR = EvalVal(RetValExp))
      Diag(DR->getLocStart(), diag::warn_ret_stack_ref)
        << DR->getDecl()->getDeclName() << RetValExp->getSourceRange();
  }
}

/// EvalAddr - EvalAddr and EvalVal are mutually recursive functions that
///  check if the expression in a return statement evaluates to an address
///  to a location on the stack.  The recursion is used to traverse the
///  AST of the return expression, with recursion backtracking when we
///  encounter a subexpression that (1) clearly does not lead to the address
///  of a stack variable or (2) is something we cannot determine leads to
///  the address of a stack variable based on such local checking.
///
///  EvalAddr processes expressions that are pointers that are used as
///  references (and not L-values).  EvalVal handles all other values.
///  At the base case of the recursion is a check for a DeclRefExpr* in
///  the refers to a stack variable.
///
///  This implementation handles:
///
///   * pointer-to-pointer casts
///   * implicit conversions from array references to pointers
///   * taking the address of fields
///   * arbitrary interplay between "&" and "*" operators
///   * pointer arithmetic from an address of a stack variable
///   * taking the address of an array element where the array is on the stack
static DeclRefExpr* EvalAddr(Expr *E) {
  // We should only be called for evaluating pointer expressions.
  assert((E->getType()->isAnyPointerType() ||
          E->getType()->isBlockPointerType() ||
          E->getType()->isObjCQualifiedIdType()) &&
         "EvalAddr only works on pointers");

  // Our "symbolic interpreter" is just a dispatch off the currently
  // viewed AST node.  We then recursively traverse the AST by calling
  // EvalAddr and EvalVal appropriately.
  switch (E->getStmtClass()) {
  case Stmt::ParenExprClass:
    // Ignore parentheses.
    return EvalAddr(cast<ParenExpr>(E)->getSubExpr());

  case Stmt::UnaryOperatorClass: {
    // The only unary operator that make sense to handle here
    // is AddrOf.  All others don't make sense as pointers.
    UnaryOperator *U = cast<UnaryOperator>(E);

    if (U->getOpcode() == UnaryOperator::AddrOf)
      return EvalVal(U->getSubExpr());
    else
      return NULL;
  }

  case Stmt::BinaryOperatorClass: {
    // Handle pointer arithmetic.  All other binary operators are not valid
    // in this context.
    BinaryOperator *B = cast<BinaryOperator>(E);
    BinaryOperator::Opcode op = B->getOpcode();

    if (op != BinaryOperator::Add && op != BinaryOperator::Sub)
      return NULL;

    Expr *Base = B->getLHS();

    // Determine which argument is the real pointer base.  It could be
    // the RHS argument instead of the LHS.
    if (!Base->getType()->isPointerType()) Base = B->getRHS();

    assert (Base->getType()->isPointerType());
    return EvalAddr(Base);
  }

  // For conditional operators we need to see if either the LHS or RHS are
  // valid DeclRefExpr*s.  If one of them is valid, we return it.
  case Stmt::ConditionalOperatorClass: {
    ConditionalOperator *C = cast<ConditionalOperator>(E);

    // Handle the GNU extension for missing LHS.
    if (Expr *lhsExpr = C->getLHS())
      if (DeclRefExpr* LHS = EvalAddr(lhsExpr))
        return LHS;

     return EvalAddr(C->getRHS());
  }

  // For casts, we need to handle conversions from arrays to
  // pointer values, and pointer-to-pointer conversions.
  case Stmt::ImplicitCastExprClass:
  case Stmt::CStyleCastExprClass:
  case Stmt::CXXFunctionalCastExprClass: {
    Expr* SubExpr = cast<CastExpr>(E)->getSubExpr();
    QualType T = SubExpr->getType();

    if (SubExpr->getType()->isPointerType() ||
        SubExpr->getType()->isBlockPointerType() ||
        SubExpr->getType()->isObjCQualifiedIdType())
      return EvalAddr(SubExpr);
    else if (T->isArrayType())
      return EvalVal(SubExpr);
    else
      return 0;
  }

  // C++ casts.  For dynamic casts, static casts, and const casts, we
  // are always converting from a pointer-to-pointer, so we just blow
  // through the cast.  In the case the dynamic cast doesn't fail (and
  // return NULL), we take the conservative route and report cases
  // where we return the address of a stack variable.  For Reinterpre
  // FIXME: The comment about is wrong; we're not always converting
  // from pointer to pointer. I'm guessing that this code should also
  // handle references to objects.
  case Stmt::CXXStaticCastExprClass:
  case Stmt::CXXDynamicCastExprClass:
  case Stmt::CXXConstCastExprClass:
  case Stmt::CXXReinterpretCastExprClass: {
      Expr *S = cast<CXXNamedCastExpr>(E)->getSubExpr();
      if (S->getType()->isPointerType() || S->getType()->isBlockPointerType())
        return EvalAddr(S);
      else
        return NULL;
  }

  // Everything else: we simply don't reason about them.
  default:
    return NULL;
  }
}


///  EvalVal - This function is complements EvalAddr in the mutual recursion.
///   See the comments for EvalAddr for more details.
static DeclRefExpr* EvalVal(Expr *E) {

  // We should only be called for evaluating non-pointer expressions, or
  // expressions with a pointer type that are not used as references but instead
  // are l-values (e.g., DeclRefExpr with a pointer type).

  // Our "symbolic interpreter" is just a dispatch off the currently
  // viewed AST node.  We then recursively traverse the AST by calling
  // EvalAddr and EvalVal appropriately.
  switch (E->getStmtClass()) {
  case Stmt::DeclRefExprClass: {
    // DeclRefExpr: the base case.  When we hit a DeclRefExpr we are looking
    //  at code that refers to a variable's name.  We check if it has local
    //  storage within the function, and if so, return the expression.
    DeclRefExpr *DR = cast<DeclRefExpr>(E);

    if (VarDecl *V = dyn_cast<VarDecl>(DR->getDecl()))
      if (V->hasLocalStorage() && !V->getType()->isReferenceType()) return DR;

    return NULL;
  }

  case Stmt::ParenExprClass:
    // Ignore parentheses.
    return EvalVal(cast<ParenExpr>(E)->getSubExpr());

  case Stmt::UnaryOperatorClass: {
    // The only unary operator that make sense to handle here
    // is Deref.  All others don't resolve to a "name."  This includes
    // handling all sorts of rvalues passed to a unary operator.
    UnaryOperator *U = cast<UnaryOperator>(E);

    if (U->getOpcode() == UnaryOperator::Deref)
      return EvalAddr(U->getSubExpr());

    return NULL;
  }

  case Stmt::ArraySubscriptExprClass: {
    // Array subscripts are potential references to data on the stack.  We
    // retrieve the DeclRefExpr* for the array variable if it indeed
    // has local storage.
    return EvalAddr(cast<ArraySubscriptExpr>(E)->getBase());
  }

  case Stmt::ConditionalOperatorClass: {
    // For conditional operators we need to see if either the LHS or RHS are
    // non-NULL DeclRefExpr's.  If one is non-NULL, we return it.
    ConditionalOperator *C = cast<ConditionalOperator>(E);

    // Handle the GNU extension for missing LHS.
    if (Expr *lhsExpr = C->getLHS())
      if (DeclRefExpr *LHS = EvalVal(lhsExpr))
        return LHS;

    return EvalVal(C->getRHS());
  }

  // Accesses to members are potential references to data on the stack.
  case Stmt::MemberExprClass: {
    MemberExpr *M = cast<MemberExpr>(E);

    // Check for indirect access.  We only want direct field accesses.
    if (!M->isArrow())
      return EvalVal(M->getBase());
    else
      return NULL;
  }

  // Everything else: we simply don't reason about them.
  default:
    return NULL;
  }
}

//===--- CHECK: Floating-Point comparisons (-Wfloat-equal) ---------------===//

/// Check for comparisons of floating point operands using != and ==.
/// Issue a warning if these are no self-comparisons, as they are not likely
/// to do what the programmer intended.
void Sema::CheckFloatComparison(SourceLocation loc, Expr* lex, Expr *rex) {
  bool EmitWarning = true;

  Expr* LeftExprSansParen = lex->IgnoreParens();
  Expr* RightExprSansParen = rex->IgnoreParens();

  // Special case: check for x == x (which is OK).
  // Do not emit warnings for such cases.
  if (DeclRefExpr* DRL = dyn_cast<DeclRefExpr>(LeftExprSansParen))
    if (DeclRefExpr* DRR = dyn_cast<DeclRefExpr>(RightExprSansParen))
      if (DRL->getDecl() == DRR->getDecl())
        EmitWarning = false;


  // Special case: check for comparisons against literals that can be exactly
  //  represented by APFloat.  In such cases, do not emit a warning.  This
  //  is a heuristic: often comparison against such literals are used to
  //  detect if a value in a variable has not changed.  This clearly can
  //  lead to false negatives.
  if (EmitWarning) {
    if (FloatingLiteral* FLL = dyn_cast<FloatingLiteral>(LeftExprSansParen)) {
      if (FLL->isExact())
        EmitWarning = false;
    } else
      if (FloatingLiteral* FLR = dyn_cast<FloatingLiteral>(RightExprSansParen)){
        if (FLR->isExact())
          EmitWarning = false;
    }
  }

  // Check for comparisons with builtin types.
  if (EmitWarning)
    if (CallExpr* CL = dyn_cast<CallExpr>(LeftExprSansParen))
      if (CL->isBuiltinCall(Context))
        EmitWarning = false;

  if (EmitWarning)
    if (CallExpr* CR = dyn_cast<CallExpr>(RightExprSansParen))
      if (CR->isBuiltinCall(Context))
        EmitWarning = false;

  // Emit the diagnostic.
  if (EmitWarning)
    Diag(loc, diag::warn_floatingpoint_eq)
      << lex->getSourceRange() << rex->getSourceRange();
}

//===--- CHECK: Integer mixed-sign comparisons (-Wsign-compare) --------===//
//===--- CHECK: Lossy implicit conversions (-Wconversion) --------------===//

namespace {

/// Structure recording the 'active' range of an integer-valued
/// expression.
struct IntRange {
  /// The number of bits active in the int.
  unsigned Width;

  /// True if the int is known not to have negative values.
  bool NonNegative;

  IntRange() {}
  IntRange(unsigned Width, bool NonNegative)
    : Width(Width), NonNegative(NonNegative)
  {}

  // Returns the range of the bool type.
  static IntRange forBoolType() {
    return IntRange(1, true);
  }

  // Returns the range of an integral type.
  static IntRange forType(ASTContext &C, QualType T) {
    return forCanonicalType(C, T->getCanonicalTypeInternal().getTypePtr());
  }

  // Returns the range of an integeral type based on its canonical
  // representation.
  static IntRange forCanonicalType(ASTContext &C, const Type *T) {
    assert(T->isCanonicalUnqualified());

    if (const VectorType *VT = dyn_cast<VectorType>(T))
      T = VT->getElementType().getTypePtr();
    if (const ComplexType *CT = dyn_cast<ComplexType>(T))
      T = CT->getElementType().getTypePtr();
    if (const EnumType *ET = dyn_cast<EnumType>(T))
      T = ET->getDecl()->getIntegerType().getTypePtr();

    const BuiltinType *BT = cast<BuiltinType>(T);
    assert(BT->isInteger());

    return IntRange(C.getIntWidth(QualType(T, 0)), BT->isUnsignedInteger());
  }

  // Returns the supremum of two ranges: i.e. their conservative merge.
  static IntRange join(const IntRange &L, const IntRange &R) {
    return IntRange(std::max(L.Width, R.Width),
                    L.NonNegative && R.NonNegative);
  }

  // Returns the infinum of two ranges: i.e. their aggressive merge.
  static IntRange meet(const IntRange &L, const IntRange &R) {
    return IntRange(std::min(L.Width, R.Width),
                    L.NonNegative || R.NonNegative);
  }
};

IntRange GetValueRange(ASTContext &C, llvm::APSInt &value, unsigned MaxWidth) {
  if (value.isSigned() && value.isNegative())
    return IntRange(value.getMinSignedBits(), false);

  if (value.getBitWidth() > MaxWidth)
    value.trunc(MaxWidth);

  // isNonNegative() just checks the sign bit without considering
  // signedness.
  return IntRange(value.getActiveBits(), true);
}

IntRange GetValueRange(ASTContext &C, APValue &result, QualType Ty,
                       unsigned MaxWidth) {
  if (result.isInt())
    return GetValueRange(C, result.getInt(), MaxWidth);

  if (result.isVector()) {
    IntRange R = GetValueRange(C, result.getVectorElt(0), Ty, MaxWidth);
    for (unsigned i = 1, e = result.getVectorLength(); i != e; ++i) {
      IntRange El = GetValueRange(C, result.getVectorElt(i), Ty, MaxWidth);
      R = IntRange::join(R, El);
    }
    return R;
  }

  if (result.isComplexInt()) {
    IntRange R = GetValueRange(C, result.getComplexIntReal(), MaxWidth);
    IntRange I = GetValueRange(C, result.getComplexIntImag(), MaxWidth);
    return IntRange::join(R, I);
  }

  // This can happen with lossless casts to intptr_t of "based" lvalues.
  // Assume it might use arbitrary bits.
  // FIXME: The only reason we need to pass the type in here is to get
  // the sign right on this one case.  It would be nice if APValue
  // preserved this.
  assert(result.isLValue());
  return IntRange(MaxWidth, Ty->isUnsignedIntegerType());
}

/// Pseudo-evaluate the given integer expression, estimating the
/// range of values it might take.
///
/// \param MaxWidth - the width to which the value will be truncated
IntRange GetExprRange(ASTContext &C, Expr *E, unsigned MaxWidth) {
  E = E->IgnoreParens();

  // Try a full evaluation first.
  Expr::EvalResult result;
  if (E->Evaluate(result, C))
    return GetValueRange(C, result.Val, E->getType(), MaxWidth);

  // I think we only want to look through implicit casts here; if the
  // user has an explicit widening cast, we should treat the value as
  // being of the new, wider type.
  if (ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(E)) {
    if (CE->getCastKind() == CastExpr::CK_NoOp)
      return GetExprRange(C, CE->getSubExpr(), MaxWidth);

    IntRange OutputTypeRange = IntRange::forType(C, CE->getType());

    bool isIntegerCast = (CE->getCastKind() == CastExpr::CK_IntegralCast);
    if (!isIntegerCast && CE->getCastKind() == CastExpr::CK_Unknown)
      isIntegerCast = CE->getSubExpr()->getType()->isIntegerType();

    // Assume that non-integer casts can span the full range of the type.
    if (!isIntegerCast)
      return OutputTypeRange;

    IntRange SubRange
      = GetExprRange(C, CE->getSubExpr(),
                     std::min(MaxWidth, OutputTypeRange.Width));

    // Bail out if the subexpr's range is as wide as the cast type.
    if (SubRange.Width >= OutputTypeRange.Width)
      return OutputTypeRange;

    // Otherwise, we take the smaller width, and we're non-negative if
    // either the output type or the subexpr is.
    return IntRange(SubRange.Width,
                    SubRange.NonNegative || OutputTypeRange.NonNegative);
  }

  if (ConditionalOperator *CO = dyn_cast<ConditionalOperator>(E)) {
    // If we can fold the condition, just take that operand.
    bool CondResult;
    if (CO->getCond()->EvaluateAsBooleanCondition(CondResult, C))
      return GetExprRange(C, CondResult ? CO->getTrueExpr()
                                        : CO->getFalseExpr(),
                          MaxWidth);

    // Otherwise, conservatively merge.
    IntRange L = GetExprRange(C, CO->getTrueExpr(), MaxWidth);
    IntRange R = GetExprRange(C, CO->getFalseExpr(), MaxWidth);
    return IntRange::join(L, R);
  }

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(E)) {
    switch (BO->getOpcode()) {

    // Boolean-valued operations are single-bit and positive.
    case BinaryOperator::LAnd:
    case BinaryOperator::LOr:
    case BinaryOperator::LT:
    case BinaryOperator::GT:
    case BinaryOperator::LE:
    case BinaryOperator::GE:
    case BinaryOperator::EQ:
    case BinaryOperator::NE:
      return IntRange::forBoolType();

    // Operations with opaque sources are black-listed.
    case BinaryOperator::PtrMemD:
    case BinaryOperator::PtrMemI:
      return IntRange::forType(C, E->getType());

    // Bitwise-and uses the *infinum* of the two source ranges.
    case BinaryOperator::And:
      return IntRange::meet(GetExprRange(C, BO->getLHS(), MaxWidth),
                            GetExprRange(C, BO->getRHS(), MaxWidth));

    // Left shift gets black-listed based on a judgement call.
    case BinaryOperator::Shl:
      return IntRange::forType(C, E->getType());

    // Right shift by a constant can narrow its left argument.
    case BinaryOperator::Shr: {
      IntRange L = GetExprRange(C, BO->getLHS(), MaxWidth);

      // If the shift amount is a positive constant, drop the width by
      // that much.
      llvm::APSInt shift;
      if (BO->getRHS()->isIntegerConstantExpr(shift, C) &&
          shift.isNonNegative()) {
        unsigned zext = shift.getZExtValue();
        if (zext >= L.Width)
          L.Width = (L.NonNegative ? 0 : 1);
        else
          L.Width -= zext;
      }

      return L;
    }

    // Comma acts as its right operand.
    case BinaryOperator::Comma:
      return GetExprRange(C, BO->getRHS(), MaxWidth);

    // Black-list pointer subtractions.
    case BinaryOperator::Sub:
      if (BO->getLHS()->getType()->isPointerType())
        return IntRange::forType(C, E->getType());
      // fallthrough
      
    default:
      break;
    }

    // Treat every other operator as if it were closed on the
    // narrowest type that encompasses both operands.
    IntRange L = GetExprRange(C, BO->getLHS(), MaxWidth);
    IntRange R = GetExprRange(C, BO->getRHS(), MaxWidth);
    return IntRange::join(L, R);
  }

  if (UnaryOperator *UO = dyn_cast<UnaryOperator>(E)) {
    switch (UO->getOpcode()) {
    // Boolean-valued operations are white-listed.
    case UnaryOperator::LNot:
      return IntRange::forBoolType();

    // Operations with opaque sources are black-listed.
    case UnaryOperator::Deref:
    case UnaryOperator::AddrOf: // should be impossible
    case UnaryOperator::OffsetOf:
      return IntRange::forType(C, E->getType());

    default:
      return GetExprRange(C, UO->getSubExpr(), MaxWidth);
    }
  }

  FieldDecl *BitField = E->getBitField();
  if (BitField) {
    llvm::APSInt BitWidthAP = BitField->getBitWidth()->EvaluateAsInt(C);
    unsigned BitWidth = BitWidthAP.getZExtValue();

    return IntRange(BitWidth, BitField->getType()->isUnsignedIntegerType());
  }

  return IntRange::forType(C, E->getType());
}

/// Checks whether the given value, which currently has the given
/// source semantics, has the same value when coerced through the
/// target semantics.
bool IsSameFloatAfterCast(const llvm::APFloat &value,
                          const llvm::fltSemantics &Src,
                          const llvm::fltSemantics &Tgt) {
  llvm::APFloat truncated = value;

  bool ignored;
  truncated.convert(Src, llvm::APFloat::rmNearestTiesToEven, &ignored);
  truncated.convert(Tgt, llvm::APFloat::rmNearestTiesToEven, &ignored);

  return truncated.bitwiseIsEqual(value);
}

/// Checks whether the given value, which currently has the given
/// source semantics, has the same value when coerced through the
/// target semantics.
///
/// The value might be a vector of floats (or a complex number).
bool IsSameFloatAfterCast(const APValue &value,
                          const llvm::fltSemantics &Src,
                          const llvm::fltSemantics &Tgt) {
  if (value.isFloat())
    return IsSameFloatAfterCast(value.getFloat(), Src, Tgt);

  if (value.isVector()) {
    for (unsigned i = 0, e = value.getVectorLength(); i != e; ++i)
      if (!IsSameFloatAfterCast(value.getVectorElt(i), Src, Tgt))
        return false;
    return true;
  }

  assert(value.isComplexFloat());
  return (IsSameFloatAfterCast(value.getComplexFloatReal(), Src, Tgt) &&
          IsSameFloatAfterCast(value.getComplexFloatImag(), Src, Tgt));
}

} // end anonymous namespace

/// \brief Implements -Wsign-compare.
///
/// \param lex the left-hand expression
/// \param rex the right-hand expression
/// \param OpLoc the location of the joining operator
/// \param Equality whether this is an "equality-like" join, which
///   suppresses the warning in some cases
void Sema::CheckSignCompare(Expr *lex, Expr *rex, SourceLocation OpLoc,
                            const PartialDiagnostic &PD, bool Equality) {
  // Don't warn if we're in an unevaluated context.
  if (ExprEvalContexts.back().Context == Unevaluated)
    return;

  // If either expression is value-dependent, don't warn. We'll get another
  // chance at instantiation time.
  if (lex->isValueDependent() || rex->isValueDependent())
    return;

  QualType lt = lex->getType(), rt = rex->getType();

  // Only warn if both operands are integral.
  if (!lt->isIntegerType() || !rt->isIntegerType())
    return;

  // In C, the width of a bitfield determines its type, and the
  // declared type only contributes the signedness.  This duplicates
  // the work that will later be done by UsualUnaryConversions.
  // Eventually, this check will be reorganized in a way that avoids
  // this duplication.
  if (!getLangOptions().CPlusPlus) {
    QualType tmp;
    tmp = Context.isPromotableBitField(lex);
    if (!tmp.isNull()) lt = tmp;
    tmp = Context.isPromotableBitField(rex);
    if (!tmp.isNull()) rt = tmp;
  }

  // The rule is that the signed operand becomes unsigned, so isolate the
  // signed operand.
  Expr *signedOperand = lex, *unsignedOperand = rex;
  QualType signedType = lt, unsignedType = rt;
  if (lt->isSignedIntegerType()) {
    if (rt->isSignedIntegerType()) return;
  } else {
    if (!rt->isSignedIntegerType()) return;
    std::swap(signedOperand, unsignedOperand);
    std::swap(signedType, unsignedType);
  }

  unsigned unsignedWidth = Context.getIntWidth(unsignedType);
  unsigned signedWidth = Context.getIntWidth(signedType);

  // If the unsigned type is strictly smaller than the signed type,
  // then (1) the result type will be signed and (2) the unsigned
  // value will fit fully within the signed type, and thus the result
  // of the comparison will be exact.
  if (signedWidth > unsignedWidth)
    return;

  // Otherwise, calculate the effective ranges.
  IntRange signedRange = GetExprRange(Context, signedOperand, signedWidth);
  IntRange unsignedRange = GetExprRange(Context, unsignedOperand, unsignedWidth);

  // We should never be unable to prove that the unsigned operand is
  // non-negative.
  assert(unsignedRange.NonNegative && "unsigned range includes negative?");

  // If the signed operand is non-negative, then the signed->unsigned
  // conversion won't change it.
  if (signedRange.NonNegative)
    return;

  // For (in)equality comparisons, if the unsigned operand is a
  // constant which cannot collide with a overflowed signed operand,
  // then reinterpreting the signed operand as unsigned will not
  // change the result of the comparison.
  if (Equality && unsignedRange.Width < unsignedWidth)
    return;

  Diag(OpLoc, PD)
    << lt << rt << lex->getSourceRange() << rex->getSourceRange();
}

/// Diagnose an implicit cast;  purely a helper for CheckImplicitConversion.
static void DiagnoseImpCast(Sema &S, Expr *E, QualType T, unsigned diag) {
  S.Diag(E->getExprLoc(), diag) << E->getType() << T << E->getSourceRange();
}

/// Implements -Wconversion.
void Sema::CheckImplicitConversion(Expr *E, QualType T) {
  // Don't diagnose in unevaluated contexts.
  if (ExprEvalContexts.back().Context == Sema::Unevaluated)
    return;

  // Don't diagnose for value-dependent expressions.
  if (E->isValueDependent())
    return;

  const Type *Source = Context.getCanonicalType(E->getType()).getTypePtr();
  const Type *Target = Context.getCanonicalType(T).getTypePtr();

  // Never diagnose implicit casts to bool.
  if (Target->isSpecificBuiltinType(BuiltinType::Bool))
    return;

  // Strip vector types.
  if (isa<VectorType>(Source)) {
    if (!isa<VectorType>(Target))
      return DiagnoseImpCast(*this, E, T, diag::warn_impcast_vector_scalar);

    Source = cast<VectorType>(Source)->getElementType().getTypePtr();
    Target = cast<VectorType>(Target)->getElementType().getTypePtr();
  }

  // Strip complex types.
  if (isa<ComplexType>(Source)) {
    if (!isa<ComplexType>(Target))
      return DiagnoseImpCast(*this, E, T, diag::warn_impcast_complex_scalar);

    Source = cast<ComplexType>(Source)->getElementType().getTypePtr();
    Target = cast<ComplexType>(Target)->getElementType().getTypePtr();
  }

  const BuiltinType *SourceBT = dyn_cast<BuiltinType>(Source);
  const BuiltinType *TargetBT = dyn_cast<BuiltinType>(Target);

  // If the source is floating point...
  if (SourceBT && SourceBT->isFloatingPoint()) {
    // ...and the target is floating point...
    if (TargetBT && TargetBT->isFloatingPoint()) {
      // ...then warn if we're dropping FP rank.

      // Builtin FP kinds are ordered by increasing FP rank.
      if (SourceBT->getKind() > TargetBT->getKind()) {
        // Don't warn about float constants that are precisely
        // representable in the target type.
        Expr::EvalResult result;
        if (E->Evaluate(result, Context)) {
          // Value might be a float, a float vector, or a float complex.
          if (IsSameFloatAfterCast(result.Val,
                     Context.getFloatTypeSemantics(QualType(TargetBT, 0)),
                     Context.getFloatTypeSemantics(QualType(SourceBT, 0))))
            return;
        }

        DiagnoseImpCast(*this, E, T, diag::warn_impcast_float_precision);
      }
      return;
    }

    // If the target is integral, always warn.
    if ((TargetBT && TargetBT->isInteger()))
      // TODO: don't warn for integer values?
      return DiagnoseImpCast(*this, E, T, diag::warn_impcast_float_integer);

    return;
  }

  if (!Source->isIntegerType() || !Target->isIntegerType())
    return;

  IntRange SourceRange = GetExprRange(Context, E, Context.getIntWidth(E->getType()));
  IntRange TargetRange = IntRange::forCanonicalType(Context, Target);

  // FIXME: also signed<->unsigned?

  if (SourceRange.Width > TargetRange.Width) {
    // People want to build with -Wshorten-64-to-32 and not -Wconversion
    // and by god we'll let them.
    if (SourceRange.Width == 64 && TargetRange.Width == 32)
      return DiagnoseImpCast(*this, E, T, diag::warn_impcast_integer_64_32);
    return DiagnoseImpCast(*this, E, T, diag::warn_impcast_integer_precision);
  }

  return;
}

// MarkLive - Mark all the blocks reachable from e as live.  Returns the total
// number of blocks just marked live.
static unsigned MarkLive(CFGBlock *e, llvm::BitVector &live) {
  unsigned count = 0;
  std::queue<CFGBlock*> workq;
  // Prep work queue
  live.set(e->getBlockID());
  ++count;
  workq.push(e);
  // Solve
  while (!workq.empty()) {
    CFGBlock *item = workq.front();
    workq.pop();
    for (CFGBlock::succ_iterator I=item->succ_begin(),
           E=item->succ_end();
         I != E;
         ++I) {
      if ((*I) && !live[(*I)->getBlockID()]) {
        live.set((*I)->getBlockID());
        ++count;
        workq.push(*I);
      }
    }
  }
  return count;
}

static SourceLocation GetUnreachableLoc(CFGBlock &b, SourceRange &R1,
                                        SourceRange &R2) {
  Stmt *S;
  unsigned sn = 0;
  R1 = R2 = SourceRange();

  top:
  if (sn < b.size())
    S = b[sn].getStmt();
  else if (b.getTerminator())
    S = b.getTerminator();
  else
    return SourceLocation();

  switch (S->getStmtClass()) {
  case Expr::BinaryOperatorClass: {
    BinaryOperator *BO = cast<BinaryOperator>(S);
    if (BO->getOpcode() == BinaryOperator::Comma) {
      if (sn+1 < b.size())
        return b[sn+1].getStmt()->getLocStart();
      CFGBlock *n = &b;
      while (1) {
        if (n->getTerminator())
          return n->getTerminator()->getLocStart();
        if (n->succ_size() != 1)
          return SourceLocation();
        n = n[0].succ_begin()[0];
        if (n->pred_size() != 1)
          return SourceLocation();
        if (!n->empty())
          return n[0][0].getStmt()->getLocStart();
      }
    }
    R1 = BO->getLHS()->getSourceRange();
    R2 = BO->getRHS()->getSourceRange();
    return BO->getOperatorLoc();
  }
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *UO = cast<UnaryOperator>(S);
    R1 = UO->getSubExpr()->getSourceRange();
    return UO->getOperatorLoc();
  }
  case Expr::CompoundAssignOperatorClass: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(S);
    R1 = CAO->getLHS()->getSourceRange();
    R2 = CAO->getRHS()->getSourceRange();
    return CAO->getOperatorLoc();
  }
  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *CO = cast<ConditionalOperator>(S);
    return CO->getQuestionLoc();
  }
  case Expr::MemberExprClass: {
    const MemberExpr *ME = cast<MemberExpr>(S);
    R1 = ME->getSourceRange();
    return ME->getMemberLoc();
  }
  case Expr::ArraySubscriptExprClass: {
    const ArraySubscriptExpr *ASE = cast<ArraySubscriptExpr>(S);
    R1 = ASE->getLHS()->getSourceRange();
    R2 = ASE->getRHS()->getSourceRange();
    return ASE->getRBracketLoc();
  }
  case Expr::CStyleCastExprClass: {
    const CStyleCastExpr *CSC = cast<CStyleCastExpr>(S);
    R1 = CSC->getSubExpr()->getSourceRange();
    return CSC->getLParenLoc();
  }
  case Expr::CXXFunctionalCastExprClass: {
    const CXXFunctionalCastExpr *CE = cast <CXXFunctionalCastExpr>(S);
    R1 = CE->getSubExpr()->getSourceRange();
    return CE->getTypeBeginLoc();
  }
  case Expr::ImplicitCastExprClass:
    ++sn;
    goto top;
  case Stmt::CXXTryStmtClass: {
    return cast<CXXTryStmt>(S)->getHandler(0)->getCatchLoc();
  }
  default: ;
  }
  R1 = S->getSourceRange();
  return S->getLocStart();
}

static SourceLocation MarkLiveTop(CFGBlock *e, llvm::BitVector &live,
                               SourceManager &SM) {
  std::queue<CFGBlock*> workq;
  // Prep work queue
  workq.push(e);
  SourceRange R1, R2;
  SourceLocation top = GetUnreachableLoc(*e, R1, R2);
  bool FromMainFile = false;
  bool FromSystemHeader = false;
  bool TopValid = false;
  if (top.isValid()) {
    FromMainFile = SM.isFromMainFile(top);
    FromSystemHeader = SM.isInSystemHeader(top);
    TopValid = true;
  }
  // Solve
  while (!workq.empty()) {
    CFGBlock *item = workq.front();
    workq.pop();
    SourceLocation c = GetUnreachableLoc(*item, R1, R2);
    if (c.isValid()
        && (!TopValid
            || (SM.isFromMainFile(c) && !FromMainFile)
            || (FromSystemHeader && !SM.isInSystemHeader(c))
            || SM.isBeforeInTranslationUnit(c, top))) {
      top = c;
      FromMainFile = SM.isFromMainFile(top);
      FromSystemHeader = SM.isInSystemHeader(top);
    }
    live.set(item->getBlockID());
    for (CFGBlock::succ_iterator I=item->succ_begin(),
           E=item->succ_end();
         I != E;
         ++I) {
      if ((*I) && !live[(*I)->getBlockID()]) {
        live.set((*I)->getBlockID());
        workq.push(*I);
      }
    }
  }
  return top;
}

static int LineCmp(const void *p1, const void *p2) {
  SourceLocation *Line1 = (SourceLocation *)p1;
  SourceLocation *Line2 = (SourceLocation *)p2;
  return !(*Line1 < *Line2);
}

namespace {
  struct ErrLoc {
    SourceLocation Loc;
    SourceRange R1;
    SourceRange R2;
    ErrLoc(SourceLocation l, SourceRange r1, SourceRange r2)
      : Loc(l), R1(r1), R2(r2) { }
  };
}

/// CheckUnreachable - Check for unreachable code.
void Sema::CheckUnreachable(AnalysisContext &AC) {
  unsigned count;
  // We avoid checking when there are errors, as the CFG won't faithfully match
  // the user's code.
  if (getDiagnostics().hasErrorOccurred())
    return;
  if (Diags.getDiagnosticLevel(diag::warn_unreachable) == Diagnostic::Ignored)
    return;

  CFG *cfg = AC.getCFG();
  if (cfg == 0)
    return;
  
  llvm::BitVector live(cfg->getNumBlockIDs());
  // Mark all live things first.
  count = MarkLive(&cfg->getEntry(), live);

  if (count == cfg->getNumBlockIDs())
    // If there are no dead blocks, we're done.
    return;

  SourceRange R1, R2;

  llvm::SmallVector<ErrLoc, 24> lines;
  bool AddEHEdges = AC.getAddEHEdges();
  // First, give warnings for blocks with no predecessors, as they
  // can't be part of a loop.
  for (CFG::iterator I = cfg->begin(), E = cfg->end(); I != E; ++I) {
    CFGBlock &b = **I;
    if (!live[b.getBlockID()]) {
      if (b.pred_begin() == b.pred_end()) {
        if (!AddEHEdges && b.getTerminator()
            && isa<CXXTryStmt>(b.getTerminator())) {
          // When not adding EH edges from calls, catch clauses
          // can otherwise seem dead.  Avoid noting them as dead.
          count += MarkLive(&b, live);
          continue;
        }
        SourceLocation c = GetUnreachableLoc(b, R1, R2);
        if (!c.isValid()) {
          // Blocks without a location can't produce a warning, so don't mark
          // reachable blocks from here as live.
          live.set(b.getBlockID());
          ++count;
          continue;
        }
        lines.push_back(ErrLoc(c, R1, R2));
        // Avoid excessive errors by marking everything reachable from here
        count += MarkLive(&b, live);
      }
    }
  }

  if (count < cfg->getNumBlockIDs()) {
    // And then give warnings for the tops of loops.
    for (CFG::iterator I = cfg->begin(), E = cfg->end(); I != E; ++I) {
      CFGBlock &b = **I;
      if (!live[b.getBlockID()])
        // Avoid excessive errors by marking everything reachable from here
        lines.push_back(ErrLoc(MarkLiveTop(&b, live, Context.getSourceManager()), SourceRange(), SourceRange()));
    }
  }

  llvm::array_pod_sort(lines.begin(), lines.end(), LineCmp);
  for (llvm::SmallVector<ErrLoc, 24>::iterator I = lines.begin(),
         E = lines.end();
       I != E;
       ++I)
    if (I->Loc.isValid())
      Diag(I->Loc, diag::warn_unreachable) << I->R1 << I->R2;
}

/// CheckFallThrough - Check that we don't fall off the end of a
/// Statement that should return a value.
///
/// \returns AlwaysFallThrough iff we always fall off the end of the statement,
/// MaybeFallThrough iff we might or might not fall off the end,
/// NeverFallThroughOrReturn iff we never fall off the end of the statement or
/// return.  We assume NeverFallThrough iff we never fall off the end of the
/// statement but we may return.  We assume that functions not marked noreturn
/// will return.
Sema::ControlFlowKind Sema::CheckFallThrough(AnalysisContext &AC) {
  CFG *cfg = AC.getCFG();
  if (cfg == 0)
    // FIXME: This should be NeverFallThrough
    return NeverFallThroughOrReturn;

  // The CFG leaves in dead things, and we don't want the dead code paths to
  // confuse us, so we mark all live things first.
  std::queue<CFGBlock*> workq;
  llvm::BitVector live(cfg->getNumBlockIDs());
  unsigned count = MarkLive(&cfg->getEntry(), live);

  bool AddEHEdges = AC.getAddEHEdges();
  if (!AddEHEdges && count != cfg->getNumBlockIDs())
    // When there are things remaining dead, and we didn't add EH edges
    // from CallExprs to the catch clauses, we have to go back and
    // mark them as live.
    for (CFG::iterator I = cfg->begin(), E = cfg->end(); I != E; ++I) {
      CFGBlock &b = **I;
      if (!live[b.getBlockID()]) {
        if (b.pred_begin() == b.pred_end()) {
          if (b.getTerminator() && isa<CXXTryStmt>(b.getTerminator()))
            // When not adding EH edges from calls, catch clauses
            // can otherwise seem dead.  Avoid noting them as dead.
            count += MarkLive(&b, live);
          continue;
        }
      }
    }

  // Now we know what is live, we check the live precessors of the exit block
  // and look for fall through paths, being careful to ignore normal returns,
  // and exceptional paths.
  bool HasLiveReturn = false;
  bool HasFakeEdge = false;
  bool HasPlainEdge = false;
  bool HasAbnormalEdge = false;
  for (CFGBlock::pred_iterator I=cfg->getExit().pred_begin(),
         E = cfg->getExit().pred_end();
       I != E;
       ++I) {
    CFGBlock& B = **I;
    if (!live[B.getBlockID()])
      continue;
    if (B.size() == 0) {
      if (B.getTerminator() && isa<CXXTryStmt>(B.getTerminator())) {
        HasAbnormalEdge = true;
        continue;
      }

      // A labeled empty statement, or the entry block...
      HasPlainEdge = true;
      continue;
    }
    Stmt *S = B[B.size()-1];
    if (isa<ReturnStmt>(S)) {
      HasLiveReturn = true;
      continue;
    }
    if (isa<ObjCAtThrowStmt>(S)) {
      HasFakeEdge = true;
      continue;
    }
    if (isa<CXXThrowExpr>(S)) {
      HasFakeEdge = true;
      continue;
    }
    if (const AsmStmt *AS = dyn_cast<AsmStmt>(S)) {
      if (AS->isMSAsm()) {
        HasFakeEdge = true;
        HasLiveReturn = true;
        continue;
      }
    }
    if (isa<CXXTryStmt>(S)) {
      HasAbnormalEdge = true;
      continue;
    }

    bool NoReturnEdge = false;
    if (CallExpr *C = dyn_cast<CallExpr>(S)) {
      if (B.succ_begin()[0] != &cfg->getExit()) {
        HasAbnormalEdge = true;
        continue;
      }
      Expr *CEE = C->getCallee()->IgnoreParenCasts();
      if (CEE->getType().getNoReturnAttr()) {
        NoReturnEdge = true;
        HasFakeEdge = true;
      } else if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CEE)) {
        ValueDecl *VD = DRE->getDecl();
        if (VD->hasAttr<NoReturnAttr>()) {
          NoReturnEdge = true;
          HasFakeEdge = true;
        }
      }
    }
    // FIXME: Add noreturn message sends.
    if (NoReturnEdge == false)
      HasPlainEdge = true;
  }
  if (!HasPlainEdge) {
    if (HasLiveReturn)
      return NeverFallThrough;
    return NeverFallThroughOrReturn;
  }
  if (HasAbnormalEdge || HasFakeEdge || HasLiveReturn)
    return MaybeFallThrough;
  // This says AlwaysFallThrough for calls to functions that are not marked
  // noreturn, that don't return.  If people would like this warning to be more
  // accurate, such functions should be marked as noreturn.
  return AlwaysFallThrough;
}

/// CheckFallThroughForFunctionDef - Check that we don't fall off the end of a
/// function that should return a value.  Check that we don't fall off the end
/// of a noreturn function.  We assume that functions and blocks not marked
/// noreturn will return.
void Sema::CheckFallThroughForFunctionDef(Decl *D, Stmt *Body,
                                          AnalysisContext &AC) {
  // FIXME: Would be nice if we had a better way to control cascading errors,
  // but for now, avoid them.  The problem is that when Parse sees:
  //   int foo() { return a; }
  // The return is eaten and the Sema code sees just:
  //   int foo() { }
  // which this code would then warn about.
  if (getDiagnostics().hasErrorOccurred())
    return;
  
  bool ReturnsVoid = false;
  bool HasNoReturn = false;
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // If the result type of the function is a dependent type, we don't know
    // whether it will be void or not, so don't 
    if (FD->getResultType()->isDependentType())
      return;
    if (FD->getResultType()->isVoidType())
      ReturnsVoid = true;
    if (FD->hasAttr<NoReturnAttr>())
      HasNoReturn = true;
  } else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (MD->getResultType()->isVoidType())
      ReturnsVoid = true;
    if (MD->hasAttr<NoReturnAttr>())
      HasNoReturn = true;
  }

  // Short circuit for compilation speed.
  if ((Diags.getDiagnosticLevel(diag::warn_maybe_falloff_nonvoid_function)
       == Diagnostic::Ignored || ReturnsVoid)
      && (Diags.getDiagnosticLevel(diag::warn_noreturn_function_has_return_expr)
          == Diagnostic::Ignored || !HasNoReturn)
      && (Diags.getDiagnosticLevel(diag::warn_suggest_noreturn_block)
          == Diagnostic::Ignored || !ReturnsVoid))
    return;
  // FIXME: Function try block
  if (CompoundStmt *Compound = dyn_cast<CompoundStmt>(Body)) {
    switch (CheckFallThrough(AC)) {
    case MaybeFallThrough:
      if (HasNoReturn)
        Diag(Compound->getRBracLoc(), diag::warn_falloff_noreturn_function);
      else if (!ReturnsVoid)
        Diag(Compound->getRBracLoc(),diag::warn_maybe_falloff_nonvoid_function);
      break;
    case AlwaysFallThrough:
      if (HasNoReturn)
        Diag(Compound->getRBracLoc(), diag::warn_falloff_noreturn_function);
      else if (!ReturnsVoid)
        Diag(Compound->getRBracLoc(), diag::warn_falloff_nonvoid_function);
      break;
    case NeverFallThroughOrReturn:
      if (ReturnsVoid && !HasNoReturn)
        Diag(Compound->getLBracLoc(), diag::warn_suggest_noreturn_function);
      break;
    case NeverFallThrough:
      break;
    }
  }
}

/// CheckFallThroughForBlock - Check that we don't fall off the end of a block
/// that should return a value.  Check that we don't fall off the end of a
/// noreturn block.  We assume that functions and blocks not marked noreturn
/// will return.
void Sema::CheckFallThroughForBlock(QualType BlockTy, Stmt *Body,
                                    AnalysisContext &AC) {
  // FIXME: Would be nice if we had a better way to control cascading errors,
  // but for now, avoid them.  The problem is that when Parse sees:
  //   int foo() { return a; }
  // The return is eaten and the Sema code sees just:
  //   int foo() { }
  // which this code would then warn about.
  if (getDiagnostics().hasErrorOccurred())
    return;
  bool ReturnsVoid = false;
  bool HasNoReturn = false;
  if (const FunctionType *FT =BlockTy->getPointeeType()->getAs<FunctionType>()){
    if (FT->getResultType()->isVoidType())
      ReturnsVoid = true;
    if (FT->getNoReturnAttr())
      HasNoReturn = true;
  }

  // Short circuit for compilation speed.
  if (ReturnsVoid
      && !HasNoReturn
      && (Diags.getDiagnosticLevel(diag::warn_suggest_noreturn_block)
          == Diagnostic::Ignored || !ReturnsVoid))
    return;
  // FIXME: Funtion try block
  if (CompoundStmt *Compound = dyn_cast<CompoundStmt>(Body)) {
    switch (CheckFallThrough(AC)) {
    case MaybeFallThrough:
      if (HasNoReturn)
        Diag(Compound->getRBracLoc(), diag::err_noreturn_block_has_return_expr);
      else if (!ReturnsVoid)
        Diag(Compound->getRBracLoc(), diag::err_maybe_falloff_nonvoid_block);
      break;
    case AlwaysFallThrough:
      if (HasNoReturn)
        Diag(Compound->getRBracLoc(), diag::err_noreturn_block_has_return_expr);
      else if (!ReturnsVoid)
        Diag(Compound->getRBracLoc(), diag::err_falloff_nonvoid_block);
      break;
    case NeverFallThroughOrReturn:
      if (ReturnsVoid)
        Diag(Compound->getLBracLoc(), diag::warn_suggest_noreturn_block);
      break;
    case NeverFallThrough:
      break;
    }
  }
}

/// CheckParmsForFunctionDef - Check that the parameters of the given
/// function are appropriate for the definition of a function. This
/// takes care of any checks that cannot be performed on the
/// declaration itself, e.g., that the types of each of the function
/// parameters are complete.
bool Sema::CheckParmsForFunctionDef(FunctionDecl *FD) {
  bool HasInvalidParm = false;
  for (unsigned p = 0, NumParams = FD->getNumParams(); p < NumParams; ++p) {
    ParmVarDecl *Param = FD->getParamDecl(p);

    // C99 6.7.5.3p4: the parameters in a parameter type list in a
    // function declarator that is part of a function definition of
    // that function shall not have incomplete type.
    //
    // This is also C++ [dcl.fct]p6.
    if (!Param->isInvalidDecl() &&
        RequireCompleteType(Param->getLocation(), Param->getType(),
                               diag::err_typecheck_decl_incomplete_type)) {
      Param->setInvalidDecl();
      HasInvalidParm = true;
    }

    // C99 6.9.1p5: If the declarator includes a parameter type list, the
    // declaration of each parameter shall include an identifier.
    if (Param->getIdentifier() == 0 &&
        !Param->isImplicit() &&
        !getLangOptions().CPlusPlus)
      Diag(Param->getLocation(), diag::err_parameter_name_omitted);
  }

  return HasInvalidParm;
}
