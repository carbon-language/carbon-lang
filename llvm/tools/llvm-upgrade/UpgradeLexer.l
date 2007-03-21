/*===-- UpgradeLexer.l - Scanner for 1.9 assembly files --------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the flex scanner for LLVM 1.9 assembly languages files.
//
//===----------------------------------------------------------------------===*/

%option prefix="Upgrade"
%option yylineno
%option nostdinit
%option never-interactive
%option batch
%option noyywrap
%option nodefault
%option 8bit
%option outfile="UpgradeLexer.cpp"
%option ecs
%option noreject
%option noyymore

%{
#include "UpgradeInternals.h"
#include "llvm/Module.h"
#include <list>
#include "UpgradeParser.h"
#include <cctype>
#include <cstdlib>

#define YY_INPUT(buf,result,max_size) \
{ \
  if (LexInput->good() && !LexInput->eof()) { \
    LexInput->read(buf,max_size); \
    result = LexInput->gcount(); \
  } else {\
    result = YY_NULL; \
  } \
}

#define YY_NEVER_INTERACTIVE 1

// Construct a token value for a non-obsolete token
#define RET_TOK(type, Enum, sym) \
  Upgradelval.type = Enum; \
  return sym

#define RET_TY(sym,NewTY,sign) \
  Upgradelval.PrimType.T = NewTY; \
  switch (sign) { \
    case 0: Upgradelval.PrimType.S.makeSignless(); break; \
    case 1: Upgradelval.PrimType.S.makeUnsigned(); break; \
    case 2: Upgradelval.PrimType.S.makeSigned(); break; \
    default: assert(0 && "Invalid sign kind"); break; \
  }\
  return sym

namespace llvm {

// TODO: All of the static identifiers are figured out by the lexer,
// these should be hashed to reduce the lexer size

// UnEscapeLexed - Run through the specified buffer and change \xx codes to the
// appropriate character.  If AllowNull is set to false, a \00 value will cause
// an exception to be thrown.
//
// If AllowNull is set to true, the return value of the function points to the
// last character of the string in memory.
//
char *UnEscapeLexed(char *Buffer, bool AllowNull) {
  char *BOut = Buffer;
  for (char *BIn = Buffer; *BIn; ) {
    if (BIn[0] == '\\' && isxdigit(BIn[1]) && isxdigit(BIn[2])) {
      char Tmp = BIn[3]; BIn[3] = 0;     // Terminate string
      *BOut = (char)strtol(BIn+1, 0, 16);  // Convert to number
      if (!AllowNull && !*BOut)
        error("String literal cannot accept \\00 escape!");

      BIn[3] = Tmp;                  // Restore character
      BIn += 3;                      // Skip over handled chars
      ++BOut;
    } else {
      *BOut++ = *BIn++;
    }
  }

  return BOut;
}

// atoull - Convert an ascii string of decimal digits into the unsigned long
// long representation... this does not have to do input error checking,
// because we know that the input will be matched by a suitable regex...
//
static uint64_t atoull(const char *Buffer) {
  uint64_t Result = 0;
  for (; *Buffer; Buffer++) {
    uint64_t OldRes = Result;
    Result *= 10;
    Result += *Buffer-'0';
    if (Result < OldRes)   // Uh, oh, overflow detected!!!
      error("constant bigger than 64 bits detected!");
  }
  return Result;
}

static uint64_t HexIntToVal(const char *Buffer) {
  uint64_t Result = 0;
  for (; *Buffer; ++Buffer) {
    uint64_t OldRes = Result;
    Result *= 16;
    char C = *Buffer;
    if (C >= '0' && C <= '9')
      Result += C-'0';
    else if (C >= 'A' && C <= 'F')
      Result += C-'A'+10;
    else if (C >= 'a' && C <= 'f')
      Result += C-'a'+10;

    if (Result < OldRes)   // Uh, oh, overflow detected!!!
      error("constant bigger than 64 bits detected!");
  }
  return Result;
}


// HexToFP - Convert the ascii string in hexidecimal format to the floating
// point representation of it.
//
static double HexToFP(const char *Buffer) {
  // Behave nicely in the face of C TBAA rules... see:
  // http://www.nullstone.com/htmls/category/aliastyp.htm
  union {
    uint64_t UI;
    double FP;
  } UIntToFP;
  UIntToFP.UI = HexIntToVal(Buffer);

  assert(sizeof(double) == sizeof(uint64_t) &&
         "Data sizes incompatible on this target!");
  return UIntToFP.FP;   // Cast Hex constant to double
}


} // End llvm namespace

using namespace llvm;

%}



/* Comments start with a ; and go till end of line */
Comment    ;.*

/* Variable(Value) identifiers start with a % sign */
VarID       [%@][-a-zA-Z$._][-a-zA-Z$._0-9]*

/* Label identifiers end with a colon */
Label       [-a-zA-Z$._0-9]+:
QuoteLabel \"[^\"]+\":

/* Quoted names can contain any character except " and \ */
StringConstant @?\"[^\"]*\"


/* [PN]Integer: match positive and negative literal integer values that
 * are preceeded by a '%' character.  These represent unnamed variable slots.
 */
EPInteger     %[0-9]+
ENInteger    %-[0-9]+


/* E[PN]Integer: match positive and negative literal integer values */
PInteger   [0-9]+
NInteger  -[0-9]+

/* FPConstant - A Floating point constant.
 */
FPConstant [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?

/* HexFPConstant - Floating point constant represented in IEEE format as a
 *  hexadecimal number for when exponential notation is not precise enough.
 */
HexFPConstant 0x[0-9A-Fa-f]+

/* HexIntConstant - Hexadecimal constant generated by the CFE to avoid forcing
 * it to deal with 64 bit numbers.
 */
HexIntConstant [us]0x[0-9A-Fa-f]+
%%

{Comment}       { /* Ignore comments for now */ }

begin           { return BEGINTOK; }
end             { return ENDTOK; }
true            { return TRUETOK;  }
false           { return FALSETOK; }
declare         { return DECLARE; }
global          { return GLOBAL; }
constant        { return CONSTANT; }
internal        { return INTERNAL; }
linkonce        { return LINKONCE; }
weak            { return WEAK; }
appending       { return APPENDING; }
dllimport       { return DLLIMPORT; }
dllexport       { return DLLEXPORT; }
extern_weak     { return EXTERN_WEAK; }
uninitialized   { return EXTERNAL; }    /* Deprecated, turn into external */
external        { return EXTERNAL; }
implementation  { return IMPLEMENTATION; }
zeroinitializer { return ZEROINITIALIZER; }
\.\.\.          { return DOTDOTDOT; }
undef           { return UNDEF; }
null            { return NULL_TOK; }
to              { return TO; }
except          { return EXCEPT; }
not             { return NOT; }  /* Deprecated, turned into XOR */
tail            { return TAIL; }
target          { return TARGET; }
triple          { return TRIPLE; }
deplibs         { return DEPLIBS; }
endian          { return ENDIAN; }
pointersize     { return POINTERSIZE; }
datalayout      { return DATALAYOUT; }
little          { return LITTLE; }
big             { return BIG; }
volatile        { return VOLATILE; }
align           { return ALIGN;  }
section         { return SECTION; }
module          { return MODULE; }
asm             { return ASM_TOK; }
sideeffect      { return SIDEEFFECT; }

cc              { return CC_TOK; }
ccc             { return CCC_TOK; }
csretcc         { return CSRETCC_TOK; }
fastcc          { return FASTCC_TOK; }
coldcc          { return COLDCC_TOK; }
x86_stdcallcc   { return X86_STDCALLCC_TOK; }
x86_fastcallcc  { return X86_FASTCALLCC_TOK; }

sbyte           { RET_TY(SBYTE,  Type::Int8Ty,  2); }
ubyte           { RET_TY(UBYTE,  Type::Int8Ty,  1); }
i8              { RET_TY(UBYTE,  Type::Int8Ty,  1); }
short           { RET_TY(SHORT,  Type::Int16Ty, 2); }
ushort          { RET_TY(USHORT, Type::Int16Ty, 1); }
i16             { RET_TY(USHORT, Type::Int16Ty, 1); }
int             { RET_TY(INT,    Type::Int32Ty, 2); }
uint            { RET_TY(UINT,   Type::Int32Ty, 1); }
i32             { RET_TY(UINT,   Type::Int32Ty, 1); }
long            { RET_TY(LONG,   Type::Int64Ty, 2); }
ulong           { RET_TY(ULONG,  Type::Int64Ty, 1); }
i64             { RET_TY(ULONG,  Type::Int64Ty, 1); }
void            { RET_TY(VOID,   Type::VoidTy,  0); }
bool            { RET_TY(BOOL,   Type::Int1Ty,  1); }
i1              { RET_TY(BOOL,   Type::Int1Ty,  1); }
float           { RET_TY(FLOAT,  Type::FloatTy, 0); }
double          { RET_TY(DOUBLE, Type::DoubleTy,0); }
label           { RET_TY(LABEL,  Type::LabelTy, 0); }
type            { return TYPE;   }
opaque          { return OPAQUE; }

add             { RET_TOK(BinaryOpVal, AddOp, ADD); }
sub             { RET_TOK(BinaryOpVal, SubOp, SUB); }
mul             { RET_TOK(BinaryOpVal, MulOp, MUL); }
div             { RET_TOK(BinaryOpVal, DivOp,  DIV); }
udiv            { RET_TOK(BinaryOpVal, UDivOp, UDIV); }
sdiv            { RET_TOK(BinaryOpVal, SDivOp, SDIV); }
fdiv            { RET_TOK(BinaryOpVal, FDivOp, FDIV); }
rem             { RET_TOK(BinaryOpVal, RemOp,  REM); }
urem            { RET_TOK(BinaryOpVal, URemOp, UREM); }
srem            { RET_TOK(BinaryOpVal, SRemOp, SREM); }
frem            { RET_TOK(BinaryOpVal, FRemOp, FREM); }
and             { RET_TOK(BinaryOpVal, AndOp, AND); }
or              { RET_TOK(BinaryOpVal, OrOp , OR ); }
xor             { RET_TOK(BinaryOpVal, XorOp, XOR); }
setne           { RET_TOK(BinaryOpVal, SetNE, SETNE); }
seteq           { RET_TOK(BinaryOpVal, SetEQ, SETEQ); }
setlt           { RET_TOK(BinaryOpVal, SetLT, SETLT); }
setgt           { RET_TOK(BinaryOpVal, SetGT, SETGT); }
setle           { RET_TOK(BinaryOpVal, SetLE, SETLE); }
setge           { RET_TOK(BinaryOpVal, SetGE, SETGE); }
shl             { RET_TOK(BinaryOpVal, ShlOp, SHL); }
shr             { RET_TOK(BinaryOpVal, ShrOp, SHR); }
lshr            { RET_TOK(BinaryOpVal, LShrOp, LSHR); }
ashr            { RET_TOK(BinaryOpVal, AShrOp, ASHR); }

icmp            { RET_TOK(OtherOpVal, ICmpOp, ICMP); }
fcmp            { RET_TOK(OtherOpVal, FCmpOp, FCMP); }

eq              { return EQ; }
ne              { return NE; }
slt             { return SLT; }
sgt             { return SGT; }
sle             { return SLE; }
sge             { return SGE; }
ult             { return ULT; }
ugt             { return UGT; }
ule             { return ULE; }
uge             { return UGE; }
oeq             { return OEQ; }
one             { return ONE; }
olt             { return OLT; }
ogt             { return OGT; }
ole             { return OLE; }
oge             { return OGE; }
ord             { return ORD; }
uno             { return UNO; }
ueq             { return UEQ; }
une             { return UNE; }

phi             { RET_TOK(OtherOpVal, PHIOp, PHI_TOK); }
call            { RET_TOK(OtherOpVal, CallOp, CALL); }
cast            { RET_TOK(CastOpVal, CastOp, CAST);  }
trunc           { RET_TOK(CastOpVal, TruncOp, TRUNC); }
zext            { RET_TOK(CastOpVal, ZExtOp , ZEXT); }
sext            { RET_TOK(CastOpVal, SExtOp, SEXT); }
fptrunc         { RET_TOK(CastOpVal, FPTruncOp, FPTRUNC); }
fpext           { RET_TOK(CastOpVal, FPExtOp, FPEXT); }
fptoui          { RET_TOK(CastOpVal, FPToUIOp, FPTOUI); }
fptosi          { RET_TOK(CastOpVal, FPToSIOp, FPTOSI); }
uitofp          { RET_TOK(CastOpVal, UIToFPOp, UITOFP); }
sitofp          { RET_TOK(CastOpVal, SIToFPOp, SITOFP); }
ptrtoint        { RET_TOK(CastOpVal, PtrToIntOp, PTRTOINT); }
inttoptr        { RET_TOK(CastOpVal, IntToPtrOp, INTTOPTR); }
bitcast         { RET_TOK(CastOpVal, BitCastOp, BITCAST); }
select          { RET_TOK(OtherOpVal, SelectOp, SELECT); }
vanext          { return VANEXT_old; }
vaarg           { return VAARG_old; }
va_arg          { RET_TOK(OtherOpVal, VAArg , VAARG); }
ret             { RET_TOK(TermOpVal, RetOp, RET); }
br              { RET_TOK(TermOpVal, BrOp, BR); }
switch          { RET_TOK(TermOpVal, SwitchOp, SWITCH); }
invoke          { RET_TOK(TermOpVal, InvokeOp, INVOKE); }
unwind          { return UNWIND; }
unreachable     { RET_TOK(TermOpVal, UnreachableOp, UNREACHABLE); }

malloc          { RET_TOK(MemOpVal, MallocOp, MALLOC); }
alloca          { RET_TOK(MemOpVal, AllocaOp, ALLOCA); }
free            { RET_TOK(MemOpVal, FreeOp, FREE); }
load            { RET_TOK(MemOpVal, LoadOp, LOAD); }
store           { RET_TOK(MemOpVal, StoreOp, STORE); }
getelementptr   { RET_TOK(MemOpVal, GetElementPtrOp, GETELEMENTPTR); }

extractelement  { RET_TOK(OtherOpVal, ExtractElementOp, EXTRACTELEMENT); }
insertelement   { RET_TOK(OtherOpVal, InsertElementOp, INSERTELEMENT); }
shufflevector   { RET_TOK(OtherOpVal, ShuffleVectorOp, SHUFFLEVECTOR); }


{VarID}         {
                  UnEscapeLexed(yytext+1);
                  Upgradelval.StrVal = strdup(yytext+1);             // Skip %
                  return VAR_ID;
                }
{Label}         {
                  yytext[strlen(yytext)-1] = 0;  // nuke colon
                  UnEscapeLexed(yytext);
                  Upgradelval.StrVal = strdup(yytext);
                  return LABELSTR;
                }
{QuoteLabel}    {
                  yytext[strlen(yytext)-2] = 0;  // nuke colon, end quote
                  UnEscapeLexed(yytext+1);
                  Upgradelval.StrVal = strdup(yytext+1);
                  return LABELSTR;
                }

{StringConstant} { // Note that we cannot unescape a string constant here!  The
                   // string constant might contain a \00 which would not be
                   // understood by the string stuff.  It is valid to make a
                   // [sbyte] c"Hello World\00" constant, for example.
                   //
                   yytext[strlen(yytext)-1] = 0;           // nuke end quote
                   Upgradelval.StrVal = strdup(yytext+1);  // Nuke start quote
                   return STRINGCONSTANT;
                 }


{PInteger}      { Upgradelval.UInt64Val = atoull(yytext); return EUINT64VAL; }
{NInteger}      {
                  uint64_t Val = atoull(yytext+1);
                  // +1:  we have bigger negative range
                  if (Val > (uint64_t)INT64_MAX+1)
                    error("Constant too large for signed 64 bits!");
                  Upgradelval.SInt64Val = -Val;
                  return ESINT64VAL;
                }
{HexIntConstant} {
                   Upgradelval.UInt64Val = HexIntToVal(yytext+3);
                   return yytext[0] == 's' ? ESINT64VAL : EUINT64VAL;
                 }

{EPInteger}     {
                  uint64_t Val = atoull(yytext+1);
                  if ((unsigned)Val != Val)
                    error("Invalid value number (too large)!");
                  Upgradelval.UIntVal = unsigned(Val);
                  return UINTVAL;
                }
{ENInteger}     {
                  uint64_t Val = atoull(yytext+2);
                  // +1:  we have bigger negative range
                  if (Val > (uint64_t)INT32_MAX+1)
                    error("Constant too large for signed 32 bits!");
                  Upgradelval.SIntVal = (int)-Val;
                  return SINTVAL;
                }

{FPConstant}    { Upgradelval.FPVal = atof(yytext); return FPVAL; }
{HexFPConstant} { Upgradelval.FPVal = HexToFP(yytext); return FPVAL; }

<<EOF>>         {
                  /* Make sure to free the internal buffers for flex when we are
                   * done reading our input!
                   */
                  yy_delete_buffer(YY_CURRENT_BUFFER);
                  return EOF;
                }

[ \r\t\n]       { /* Ignore whitespace */ }
.               { return yytext[0]; }

%%
