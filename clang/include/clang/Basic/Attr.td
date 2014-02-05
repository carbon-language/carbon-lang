//==--- Attr.td - attribute definitions -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// An attribute's subject is whatever it appertains to. In this file, it is
// more accurately a list of things that an attribute can appertain to. All
// Decls and Stmts are possibly AttrSubjects (even though the syntax may not
// allow attributes on a given Decl or Stmt).
class AttrSubject;

include "clang/Basic/DeclNodes.td"
include "clang/Basic/StmtNodes.td"

// A subset-subject is an AttrSubject constrained to operate only on some subset
// of that subject.
//
// The code fragment is a boolean expression that will confirm that the subject
// meets the requirements; the subject will have the name S, and will have the
// type specified by the base. It should be a simple boolean expression.
class SubsetSubject<AttrSubject base, code check> : AttrSubject {
  AttrSubject Base = base;
  code CheckCode = check;
}

// This is the type of a variable which C++11 allows alignas(...) to appertain
// to.
def NormalVar : SubsetSubject<Var,
                              [{S->getStorageClass() != VarDecl::Register &&
                                S->getKind() != Decl::ImplicitParam &&
                                S->getKind() != Decl::ParmVar &&
                                S->getKind() != Decl::NonTypeTemplateParm}]>;
def NonBitField : SubsetSubject<Field,
                                [{!S->isBitField()}]>;

def ObjCInstanceMethod : SubsetSubject<ObjCMethod,
                                       [{S->isInstanceMethod()}]>;

def ObjCInterfaceDeclInitMethod : SubsetSubject<ObjCMethod,
                               [{S->getMethodFamily() == OMF_init &&
                                 isa<ObjCInterfaceDecl>(S->getDeclContext())}]>;

def Struct : SubsetSubject<Record,
                           [{!S->isUnion()}]>;

def TLSVar : SubsetSubject<Var,
                           [{S->getTLSKind() != 0}]>;

def SharedVar : SubsetSubject<Var,
                              [{S->hasGlobalStorage() && !S->getTLSKind()}]>;

def GlobalVar : SubsetSubject<Var,
                             [{S->hasGlobalStorage()}]>;

// FIXME: this hack is needed because DeclNodes.td defines the base Decl node
// type to be a class, not a definition. This makes it impossible to create an
// attribute subject which accepts a Decl. Normally, this is not a problem,
// because the attribute can have no Subjects clause to accomplish this. But in
// the case of a SubsetSubject, there's no way to express it without this hack.
def DeclBase : AttrSubject;
def FunctionLike : SubsetSubject<DeclBase,
                                  [{S->getFunctionType(false) != NULL}]>;

// HasFunctionProto is a more strict version of FunctionLike, so it should
// never be specified in a Subjects list along with FunctionLike (due to the
// inclusive nature of subject testing).
def HasFunctionProto : SubsetSubject<DeclBase,
                                     [{(S->getFunctionType(true) != NULL &&
                              isa<FunctionProtoType>(S->getFunctionType())) ||
                                       isa<ObjCMethodDecl>(S) ||
                                       isa<BlockDecl>(S)}]>;

// A single argument to an attribute
class Argument<string name, bit optional> {
  string Name = name;
  bit Optional = optional;
}

class BoolArgument<string name, bit opt = 0> : Argument<name, opt>;
class IdentifierArgument<string name, bit opt = 0> : Argument<name, opt>;
class IntArgument<string name, bit opt = 0> : Argument<name, opt>;
class StringArgument<string name, bit opt = 0> : Argument<name, opt>;
class ExprArgument<string name, bit opt = 0> : Argument<name, opt>;
class FunctionArgument<string name, bit opt = 0> : Argument<name, opt>;
class TypeArgument<string name, bit opt = 0> : Argument<name, opt>;
class UnsignedArgument<string name, bit opt = 0> : Argument<name, opt>;
class VariadicUnsignedArgument<string name> : Argument<name, 1>;
class VariadicExprArgument<string name> : Argument<name, 1>;

// A version of the form major.minor[.subminor].
class VersionArgument<string name, bit opt = 0> : Argument<name, opt>;

// This one's a doozy, so it gets its own special type
// It can be an unsigned integer, or a type. Either can
// be dependent.
class AlignedArgument<string name, bit opt = 0> : Argument<name, opt>;

// An integer argument with a default value
class DefaultIntArgument<string name, int default> : IntArgument<name, 1> {
  int Default = default;
}

// This argument is more complex, it includes the enumerator type name,
// a list of strings to accept, and a list of enumerators to map them to.
class EnumArgument<string name, string type, list<string> values,
                   list<string> enums, bit opt = 0> : Argument<name, opt> {
  string Type = type;
  list<string> Values = values;
  list<string> Enums = enums;
}

// FIXME: There should be a VariadicArgument type that takes any other type
//        of argument and generates the appropriate type.
class VariadicEnumArgument<string name, string type, list<string> values,
                           list<string> enums> : Argument<name, 1>  {
  string Type = type;
  list<string> Values = values;
  list<string> Enums = enums;
}

// This handles one spelling of an attribute.
class Spelling<string name, string variety> {
  string Name = name;
  string Variety = variety;
  bit KnownToGCC;
}

class GNU<string name> : Spelling<name, "GNU">;
class Declspec<string name> : Spelling<name, "Declspec">;
class CXX11<string namespace, string name> : Spelling<name, "CXX11"> {
  string Namespace = namespace;
}
class Keyword<string name> : Spelling<name, "Keyword">;

// The GCC spelling implies GNU<name, "GNU"> and CXX11<"gnu", name> and also
// sets KnownToGCC to 1. This spelling should be used for any GCC-compatible
// attributes.
class GCC<string name> : Spelling<name, "GCC"> {
  let KnownToGCC = 1;
}

class Accessor<string name, list<Spelling> spellings> {
  string Name = name;
  list<Spelling> Spellings = spellings;
}

class SubjectDiag<bit warn> {
  bit Warn = warn;
}
def WarnDiag : SubjectDiag<1>;
def ErrorDiag : SubjectDiag<0>;

class SubjectList<list<AttrSubject> subjects, SubjectDiag diag = WarnDiag,
                  string customDiag = ""> {
  list<AttrSubject> Subjects = subjects;
  SubjectDiag Diag = diag;
  string CustomDiag = customDiag;
}

class LangOpt<string name> {
  string Name = name;
}
def MicrosoftExt : LangOpt<"MicrosoftExt">;
def Borland : LangOpt<"Borland">;
def CUDA : LangOpt<"CUDA">;

// Defines targets for target-specific attributes. The list of strings should
// specify architectures for which the target applies, based off the ArchType
// enumeration in Triple.h.
class TargetArch<list<string> arches> {
  list<string> Arches = arches;
  list<string> OSes;
}
def TargetARM : TargetArch<["arm", "thumb"]>;
def TargetMSP430 : TargetArch<["msp430"]>;
def TargetX86 : TargetArch<["x86"]>;
def TargetX86Win : TargetArch<["x86", "x86_64"]> {
  let OSes = ["Win32", "MinGW32"];
}
def TargetMips : TargetArch<["mips", "mipsel"]>;

class Attr {
  // The various ways in which an attribute can be spelled in source
  list<Spelling> Spellings;
  // The things to which an attribute can appertain
  SubjectList Subjects;
  // The arguments allowed on an attribute
  list<Argument> Args = [];
  // Accessors which should be generated for the attribute.
  list<Accessor> Accessors = [];
  // Set to true for attributes with arguments which require delayed parsing.
  bit LateParsed = 0;
  // Set to false to prevent an attribute from being propagated from a template
  // to the instantiation.
  bit Clone = 1;
  // Set to true for attributes which must be instantiated within templates
  bit TemplateDependent = 0;
  // Set to true for attributes that have a corresponding AST node.
  bit ASTNode = 1;
  // Set to true for attributes which have handler in Sema.
  bit SemaHandler = 1;
  // Set to true for attributes that are completely ignored.
  bit Ignored = 0;
  // Set to true if the attribute's parsing does not match its semantic
  // content. Eg) It parses 3 args, but semantically takes 4 args.  Opts out of
  // common attribute error checking.
  bit HasCustomParsing = 0;
  // Set to true if all of the attribute's arguments should be parsed in an
  // unevaluated context.
  bit ParseArgumentsAsUnevaluated = 0;
  // Set to true if this attribute can be duplicated on a subject when merging
  // attributes. By default, attributes are not merged.
  bit DuplicatesAllowedWhileMerging = 0;
  // Lists language options, one of which is required to be true for the
  // attribute to be applicable. If empty, no language options are required.
  list<LangOpt> LangOpts = [];
  // Any additional text that should be included verbatim in the class.
  code AdditionalMembers = [{}];
}

/// A type attribute is not processed on a declaration or a statement.
class TypeAttr : Attr {
  // By default, type attributes do not get an AST node.
  let ASTNode = 0;
}

/// An inheritable attribute is inherited by later redeclarations.
class InheritableAttr : Attr;

/// A target-specific attribute.  This class is meant to be used as a mixin
/// with InheritableAttr or Attr depending on the attribute's needs.
class TargetSpecificAttr<TargetArch target> {
  TargetArch Target = target;
  // Attributes are generally required to have unique spellings for their names
  // so that the parser can determine what kind of attribute it has parsed.
  // However, target-specific attributes are special in that the attribute only
  // "exists" for a given target. So two target-specific attributes can share
  // the same name when they exist in different targets. To support this, a
  // Kind can be explicitly specified for a target-specific attribute. This
  // corresponds to the AttributeList::AT_* enum that is generated and it
  // should contain a shared value between the attributes.
  //
  // Target-specific attributes which use this feature should ensure that the
  // spellings match exactly betweeen the attributes, and if the arguments or
  // subjects differ, should specify HasCustomParsing = 1 and implement their
  // own parsing and semantic handling requirements as-needed.
  string ParseKind;
}

/// An inheritable parameter attribute is inherited by later
/// redeclarations, even when it's written on a parameter.
class InheritableParamAttr : InheritableAttr;

/// An ignored attribute, which we parse but discard with no checking.
class IgnoredAttr : Attr {
  let Ignored = 1;
  let ASTNode = 0;
  let SemaHandler = 0;
}

//
// Attributes begin here
//

def AddressSpace : TypeAttr {
  let Spellings = [GNU<"address_space">];
  let Args = [IntArgument<"AddressSpace">];
}

def Alias : Attr {
  let Spellings = [GCC<"alias">];
  let Args = [StringArgument<"Aliasee">];
}

def Aligned : InheritableAttr {
  let Spellings = [GCC<"aligned">, Declspec<"align">, Keyword<"alignas">,
                   Keyword<"_Alignas">];
//  let Subjects = SubjectList<[NonBitField, NormalVar, Tag]>;
  let Args = [AlignedArgument<"Alignment", 1>];
  let Accessors = [Accessor<"isGNU", [GCC<"aligned">]>,
                   Accessor<"isC11", [Keyword<"_Alignas">]>,
                   Accessor<"isAlignas", [Keyword<"alignas">,
                                          Keyword<"_Alignas">]>,
                   Accessor<"isDeclspec",[Declspec<"align">]>];
}

def AlignMac68k : InheritableAttr {
  // This attribute has no spellings as it is only ever created implicitly.
  let Spellings = [];
  let SemaHandler = 0;
}

def AlwaysInline : InheritableAttr {
  let Spellings = [GCC<"always_inline">];
  let Subjects = SubjectList<[Function]>;
}

def TLSModel : InheritableAttr {
  let Spellings = [GCC<"tls_model">];
  let Subjects = SubjectList<[TLSVar], ErrorDiag, "ExpectedTLSVar">;
  let Args = [StringArgument<"Model">];
}

def AnalyzerNoReturn : InheritableAttr {
  let Spellings = [GNU<"analyzer_noreturn">];
}

def Annotate : InheritableParamAttr {
  let Spellings = [GNU<"annotate">];
  let Args = [StringArgument<"Annotation">];
}

def ARMInterrupt : InheritableAttr, TargetSpecificAttr<TargetARM> {
  // NOTE: If you add any additional spellings, MSP430Interrupt's spellings
  // must match.
  let Spellings = [GNU<"interrupt">];
  let Args = [EnumArgument<"Interrupt", "InterruptType",
                           ["IRQ", "FIQ", "SWI", "ABORT", "UNDEF", ""],
                           ["IRQ", "FIQ", "SWI", "ABORT", "UNDEF", "Generic"],
                           1>];
  let ParseKind = "Interrupt";
  let HasCustomParsing = 1;
}

def AsmLabel : InheritableAttr {
  let Spellings = [Keyword<"asm">, Keyword<"__asm__">];
  let Args = [StringArgument<"Label">];
  let SemaHandler = 0;
}

def Availability : InheritableAttr {
  let Spellings = [GNU<"availability">];
  let Args = [IdentifierArgument<"platform">, VersionArgument<"introduced">,
              VersionArgument<"deprecated">, VersionArgument<"obsoleted">,
              BoolArgument<"unavailable">, StringArgument<"message">];
  let AdditionalMembers =
[{static llvm::StringRef getPrettyPlatformName(llvm::StringRef Platform) {
    return llvm::StringSwitch<llvm::StringRef>(Platform)
             .Case("ios", "iOS")
             .Case("macosx", "OS X")
             .Default(llvm::StringRef());
} }];
  let HasCustomParsing = 1;
  let DuplicatesAllowedWhileMerging = 1;
//  let Subjects = SubjectList<[Named]>;
}

def Blocks : InheritableAttr {
  let Spellings = [GNU<"blocks">];
  let Args = [EnumArgument<"Type", "BlockType", ["byref"], ["ByRef"]>];
}

def Bounded : IgnoredAttr {
  let Spellings = [GNU<"bounded">];
}

def CarriesDependency : InheritableParamAttr {
  let Spellings = [GNU<"carries_dependency">, CXX11<"","carries_dependency">,
                   CXX11<"std","carries_dependency">];
  let Subjects = SubjectList<[ParmVar, ObjCMethod, Function], ErrorDiag>;
}

def CDecl : InheritableAttr {
  let Spellings = [GCC<"cdecl">, Keyword<"__cdecl">, Keyword<"_cdecl">];
//  let Subjects = [Function, ObjCMethod];
}

// cf_audited_transfer indicates that the given function has been
// audited and has been marked with the appropriate cf_consumed and
// cf_returns_retained attributes.  It is generally applied by
// '#pragma clang arc_cf_code_audited' rather than explicitly.
def CFAuditedTransfer : InheritableAttr {
  let Spellings = [GNU<"cf_audited_transfer">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

// cf_unknown_transfer is an explicit opt-out of cf_audited_transfer.
// It indicates that the function has unknown or unautomatable
// transfer semantics.
def CFUnknownTransfer : InheritableAttr {
  let Spellings = [GNU<"cf_unknown_transfer">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def CFReturnsRetained : InheritableAttr {
  let Spellings = [GNU<"cf_returns_retained">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
}

def CFReturnsNotRetained : InheritableAttr {
  let Spellings = [GNU<"cf_returns_not_retained">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
}

def CFConsumed : InheritableParamAttr {
  let Spellings = [GNU<"cf_consumed">];
  let Subjects = SubjectList<[ParmVar]>;
}

def Cleanup : InheritableAttr {
  let Spellings = [GCC<"cleanup">];
  let Args = [FunctionArgument<"FunctionDecl">];
  let Subjects = SubjectList<[Var]>;
}

def Cold : InheritableAttr {
  let Spellings = [GCC<"cold">];
  let Subjects = SubjectList<[Function]>;
}

def Common : InheritableAttr {
  let Spellings = [GCC<"common">];
  let Subjects = SubjectList<[Var]>;
}

def Const : InheritableAttr {
  let Spellings = [GCC<"const">, GCC<"__const">];
}

def Constructor : InheritableAttr {
  let Spellings = [GCC<"constructor">];
  let Args = [DefaultIntArgument<"Priority", 65535>];
  let Subjects = SubjectList<[Function]>;
}

def CUDAConstant : InheritableAttr {
  let Spellings = [GNU<"constant">];
  let Subjects = SubjectList<[Var]>;
  let LangOpts = [CUDA];
}

def CUDADevice : InheritableAttr {
  let Spellings = [GNU<"device">];
  let Subjects = SubjectList<[Function, Var]>;
  let LangOpts = [CUDA];
}

def CUDAGlobal : InheritableAttr {
  let Spellings = [GNU<"global">];
  let Subjects = SubjectList<[Function]>;
  let LangOpts = [CUDA];
}

def CUDAHost : InheritableAttr {
  let Spellings = [GNU<"host">];
  let Subjects = SubjectList<[Function]>;
  let LangOpts = [CUDA];
}

def CUDALaunchBounds : InheritableAttr {
  let Spellings = [GNU<"launch_bounds">];
  let Args = [IntArgument<"MaxThreads">, DefaultIntArgument<"MinBlocks", 0>];
  let LangOpts = [CUDA];
  let Subjects = SubjectList<[ObjCMethod, FunctionLike], WarnDiag,
                             "ExpectedFunctionOrMethod">;
  // An AST node is created for this attribute, but is not used by other parts
  // of the compiler. However, this node needs to exist in the AST because
  // non-LLVM backends may be relying on the attribute's presence.
}

def CUDAShared : InheritableAttr {
  let Spellings = [GNU<"shared">];
  let Subjects = SubjectList<[Var]>;
  let LangOpts = [CUDA];
}

def C11NoReturn : InheritableAttr {
  let Spellings = [Keyword<"_Noreturn">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let SemaHandler = 0;
}

def CXX11NoReturn : InheritableAttr {
  let Spellings = [CXX11<"","noreturn">, CXX11<"std","noreturn">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def OpenCLKernel : InheritableAttr {
  let Spellings = [Keyword<"__kernel">, Keyword<"kernel">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

// This attribute is both a type attribute, and a declaration attribute (for
// parameter variables).
def OpenCLImageAccess : Attr {
  let Spellings = [Keyword<"__read_only">, Keyword<"read_only">,
                   Keyword<"__write_only">, Keyword<"write_only">,
                   Keyword<"__read_write">, Keyword<"read_write">];
  let Subjects = SubjectList<[ParmVar], ErrorDiag>;
  let Accessors = [Accessor<"isReadOnly", [Keyword<"__read_only">,
                                           Keyword<"read_only">]>,
                   Accessor<"isReadWrite", [Keyword<"__read_write">,
                                            Keyword<"read_write">]>,
                   Accessor<"isWriteOnly", [Keyword<"__write_only">,
                                            Keyword<"write_only">]>];
}

def OpenCLPrivateAddressSpace : TypeAttr {
  let Spellings = [Keyword<"__private">, Keyword<"private">];
}

def OpenCLGlobalAddressSpace : TypeAttr {
  let Spellings = [Keyword<"__global">, Keyword<"global">];
}

def OpenCLLocalAddressSpace : TypeAttr {
  let Spellings = [Keyword<"__local">, Keyword<"local">];
}

def OpenCLConstantAddressSpace : TypeAttr {
  let Spellings = [Keyword<"__constant">, Keyword<"constant">];
}

def Deprecated : InheritableAttr {
  let Spellings = [GCC<"deprecated">, Declspec<"deprecated">,
                   CXX11<"","deprecated">];
  let Args = [StringArgument<"Message", 1>];
}

def Destructor : InheritableAttr {
  let Spellings = [GCC<"destructor">];
  let Args = [DefaultIntArgument<"Priority", 65535>];
  let Subjects = SubjectList<[Function]>;
}

def EnableIf : InheritableAttr {
  let Spellings = [GNU<"enable_if">];
  let Subjects = SubjectList<[Function]>;
  let Args = [ExprArgument<"Cond">, StringArgument<"Message">];
  let TemplateDependent = 1;
}

def ExtVectorType : Attr {
  let Spellings = [GNU<"ext_vector_type">];
  let Subjects = SubjectList<[TypedefName], ErrorDiag>;
  let Args = [ExprArgument<"NumElements">];
  let ASTNode = 0;
}

def FallThrough : Attr {
  let Spellings = [CXX11<"clang", "fallthrough">];
//  let Subjects = [NullStmt];
}

def FastCall : InheritableAttr {
  let Spellings = [GCC<"fastcall">, Keyword<"__fastcall">,
                   Keyword<"_fastcall">];
//  let Subjects = [Function, ObjCMethod];
}

def Final : InheritableAttr {
  let Spellings = [Keyword<"final">, Keyword<"sealed">];
  let Accessors = [Accessor<"isSpelledAsSealed", [Keyword<"sealed">]>];
  let SemaHandler = 0;
}

def MinSize : InheritableAttr {
  let Spellings = [GNU<"minsize">];
  let Subjects = SubjectList<[Function, ObjCMethod], ErrorDiag>;
}

def Format : InheritableAttr {
  let Spellings = [GCC<"format">];
  let Args = [IdentifierArgument<"Type">, IntArgument<"FormatIdx">,
              IntArgument<"FirstArg">];
  let Subjects = SubjectList<[ObjCMethod, Block, HasFunctionProto], WarnDiag,
                             "ExpectedFunction">;
}

def FormatArg : InheritableAttr {
  let Spellings = [GCC<"format_arg">];
  let Args = [IntArgument<"FormatIdx">];
  let Subjects = SubjectList<[ObjCMethod, HasFunctionProto], WarnDiag,
                             "ExpectedFunction">;
}

def GNUInline : InheritableAttr {
  let Spellings = [GCC<"gnu_inline">];
  let Subjects = SubjectList<[Function]>;
}

def Hot : InheritableAttr {
  let Spellings = [GCC<"hot">];
  let Subjects = SubjectList<[Function]>;
  // An AST node is created for this attribute, but not actually used beyond
  // semantic checking for mutual exclusion with the Cold attribute.
}

def IBAction : InheritableAttr {
  let Spellings = [GNU<"ibaction">];
  let Subjects = SubjectList<[ObjCInstanceMethod], WarnDiag,
                             "ExpectedObjCInstanceMethod">;
  // An AST node is created for this attribute, but is not used by other parts
  // of the compiler. However, this node needs to exist in the AST because
  // external tools rely on it.
}

def IBOutlet : InheritableAttr {
  let Spellings = [GNU<"iboutlet">];
//  let Subjects = [ObjCIvar, ObjCProperty];
}

def IBOutletCollection : InheritableAttr {
  let Spellings = [GNU<"iboutletcollection">];
  let Args = [TypeArgument<"Interface", 1>];
//  let Subjects = [ObjCIvar, ObjCProperty];
}

def Malloc : InheritableAttr {
  let Spellings = [GCC<"malloc">];
//  let Subjects = [Function];
}

def MaxFieldAlignment : InheritableAttr {
  // This attribute has no spellings as it is only ever created implicitly.
  let Spellings = [];
  let Args = [UnsignedArgument<"Alignment">];
  let SemaHandler = 0;
}

def MayAlias : InheritableAttr {
  // FIXME: this is a type attribute in GCC, but a declaration attribute here.
  let Spellings = [GCC<"may_alias">];
}

def MSABI : InheritableAttr {
  let Spellings = [GCC<"ms_abi">];
//  let Subjects = [Function, ObjCMethod];
}

def MSP430Interrupt : InheritableAttr, TargetSpecificAttr<TargetMSP430> {
  // NOTE: If you add any additional spellings, ARMInterrupt's spellings must
  // match.
  let Spellings = [GNU<"interrupt">];
  let Args = [UnsignedArgument<"Number">];
  let ParseKind = "Interrupt";
  let HasCustomParsing = 1;
}

def Mips16 : InheritableAttr, TargetSpecificAttr<TargetMips> {
  let Spellings = [GCC<"mips16">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def Mode : Attr {
  let Spellings = [GCC<"mode">];
  let Args = [IdentifierArgument<"Mode">];
}

def Naked : InheritableAttr {
  let Spellings = [GCC<"naked">, Declspec<"naked">];
  let Subjects = SubjectList<[Function]>;
}

def NeonPolyVectorType : TypeAttr {
  let Spellings = [GNU<"neon_polyvector_type">];
  let Args = [IntArgument<"NumElements">];
}

def NeonVectorType : TypeAttr {
  let Spellings = [GNU<"neon_vector_type">];
  let Args = [IntArgument<"NumElements">];
}

def ReturnsTwice : InheritableAttr {
  let Spellings = [GCC<"returns_twice">];
  let Subjects = SubjectList<[Function]>;
}

def NoCommon : InheritableAttr {
  let Spellings = [GCC<"nocommon">];
  let Subjects = SubjectList<[Var]>;
}

def NoDebug : InheritableAttr {
  let Spellings = [GNU<"nodebug">];
}

def NoInline : InheritableAttr {
  let Spellings = [GCC<"noinline">, Declspec<"noinline">];
  let Subjects = SubjectList<[Function]>;
}

def NoMips16 : InheritableAttr, TargetSpecificAttr<TargetMips> {
  let Spellings = [GCC<"nomips16">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def NonNull : InheritableAttr {
  let Spellings = [GCC<"nonnull">];
  let Subjects = SubjectList<[ObjCMethod, HasFunctionProto, ParmVar], WarnDiag,
                             "ExpectedFunctionMethodOrParameter">;
  let Args = [VariadicUnsignedArgument<"Args">];
  let AdditionalMembers =
[{bool isNonNull(unsigned idx) const {
    for (args_iterator i = args_begin(), e = args_end();
         i != e; ++i)
      if (*i == idx)
        return true;
    return false;
  } }];
}

def ReturnsNonNull : InheritableAttr {
  let Spellings = [GCC<"returns_nonnull">];
  let Subjects = SubjectList<[ObjCMethod, HasFunctionProto], WarnDiag,
                             "ExpectedFunctionOrMethod">;
}

def NoReturn : InheritableAttr {
  let Spellings = [GCC<"noreturn">, Declspec<"noreturn">];
  // FIXME: Does GCC allow this on the function instead?
}

def NoInstrumentFunction : InheritableAttr {
  let Spellings = [GCC<"no_instrument_function">];
  let Subjects = SubjectList<[Function]>;
}

def NoThrow : InheritableAttr {
  let Spellings = [GCC<"nothrow">, Declspec<"nothrow">];
}

def ObjCBridge : InheritableAttr {
  let Spellings = [GNU<"objc_bridge">];
  let Subjects = SubjectList<[Record], ErrorDiag>;
  let Args = [IdentifierArgument<"BridgedType">];
}

def ObjCBridgeMutable : InheritableAttr {
  let Spellings = [GNU<"objc_bridge_mutable">];
  let Subjects = SubjectList<[Record], ErrorDiag>;
  let Args = [IdentifierArgument<"BridgedType">];
}

def ObjCBridgeRelated : InheritableAttr {
  let Spellings = [GNU<"objc_bridge_related">];
  let Subjects = SubjectList<[Record], ErrorDiag>;
  let Args = [IdentifierArgument<"RelatedClass">,
          IdentifierArgument<"ClassMethod">,
          IdentifierArgument<"InstanceMethod">];
 let HasCustomParsing = 1;
}

def NSReturnsRetained : InheritableAttr {
  let Spellings = [GNU<"ns_returns_retained">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
}

def NSReturnsNotRetained : InheritableAttr {
  let Spellings = [GNU<"ns_returns_not_retained">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
}

def NSReturnsAutoreleased : InheritableAttr {
  let Spellings = [GNU<"ns_returns_autoreleased">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
}

def NSConsumesSelf : InheritableAttr {
  let Spellings = [GNU<"ns_consumes_self">];
  let Subjects = SubjectList<[ObjCMethod]>;
}

def NSConsumed : InheritableParamAttr {
  let Spellings = [GNU<"ns_consumed">];
  let Subjects = SubjectList<[ParmVar]>;
}

def ObjCException : InheritableAttr {
  let Spellings = [GNU<"objc_exception">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
}

def ObjCMethodFamily : InheritableAttr {
  let Spellings = [GNU<"objc_method_family">];
  let Subjects = SubjectList<[ObjCMethod], ErrorDiag>;
  let Args = [EnumArgument<"Family", "FamilyKind",
               ["none", "alloc", "copy", "init", "mutableCopy", "new"],
               ["OMF_None", "OMF_alloc", "OMF_copy", "OMF_init",
                "OMF_mutableCopy", "OMF_new"]>];
}

def ObjCNSObject : InheritableAttr {
  let Spellings = [GNU<"NSObject">];
}

def ObjCPreciseLifetime : InheritableAttr {
  let Spellings = [GNU<"objc_precise_lifetime">];
  let Subjects = SubjectList<[Var], ErrorDiag>;
}

def ObjCReturnsInnerPointer : InheritableAttr {
  let Spellings = [GNU<"objc_returns_inner_pointer">];
  let Subjects = SubjectList<[ObjCMethod, ObjCProperty], ErrorDiag>;
}

def ObjCRequiresSuper : InheritableAttr {
  let Spellings = [GNU<"objc_requires_super">];
  let Subjects = SubjectList<[ObjCMethod], ErrorDiag>;
}

def ObjCRootClass : InheritableAttr {
  let Spellings = [GNU<"objc_root_class">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
}

def ObjCExplicitProtocolImpl : InheritableAttr {
  let Spellings = [GNU<"objc_protocol_requires_explicit_implementation">];
  let Subjects = SubjectList<[ObjCProtocol], ErrorDiag>;
}

def ObjCDesignatedInitializer : Attr {
  let Spellings = [GNU<"objc_designated_initializer">];
  let Subjects = SubjectList<[ObjCInterfaceDeclInitMethod], ErrorDiag,
                             "ExpectedObjCInterfaceDeclInitMethod">;
}

def Overloadable : Attr {
  let Spellings = [GNU<"overloadable">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def Override : InheritableAttr { 
  let Spellings = [Keyword<"override">];
  let SemaHandler = 0;
}

def Ownership : InheritableAttr {
  let Spellings = [GNU<"ownership_holds">, GNU<"ownership_returns">,
                   GNU<"ownership_takes">];
  let Accessors = [Accessor<"isHolds", [GNU<"ownership_holds">]>,
                   Accessor<"isReturns", [GNU<"ownership_returns">]>,
                   Accessor<"isTakes", [GNU<"ownership_takes">]>];
  let AdditionalMembers = [{
    enum OwnershipKind { Holds, Returns, Takes };
    OwnershipKind getOwnKind() const {
      return isHolds() ? Holds :
             isTakes() ? Takes :
             Returns;
    }
  }];
  let Args = [IdentifierArgument<"Module">, VariadicUnsignedArgument<"Args">];
  let Subjects = SubjectList<[HasFunctionProto], WarnDiag, "ExpectedFunction">;
}

def Packed : InheritableAttr {
  let Spellings = [GCC<"packed">];
//  let Subjects = [Tag, Field];
}

def PnaclCall : InheritableAttr {
  let Spellings = [GNU<"pnaclcall">];
//  let Subjects = [Function, ObjCMethod];
}

def IntelOclBicc : InheritableAttr {
  let Spellings = [GNU<"intel_ocl_bicc">];
//  let Subjects = [Function, ObjCMethod];
}

def Pcs : InheritableAttr {
  let Spellings = [GCC<"pcs">];
  let Args = [EnumArgument<"PCS", "PCSType",
                           ["aapcs", "aapcs-vfp"],
                           ["AAPCS", "AAPCS_VFP"]>];
//  let Subjects = [Function, ObjCMethod];
}

def Pure : InheritableAttr {
  let Spellings = [GCC<"pure">];
}

def Regparm : TypeAttr {
  let Spellings = [GCC<"regparm">];
  let Args = [UnsignedArgument<"NumParams">];
  let ASTNode = 0;
}

def ReqdWorkGroupSize : InheritableAttr {
  let Spellings = [GNU<"reqd_work_group_size">];
  let Args = [UnsignedArgument<"XDim">, UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def WorkGroupSizeHint :  InheritableAttr {
  let Spellings = [GNU<"work_group_size_hint">];
  let Args = [UnsignedArgument<"XDim">, 
              UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def InitPriority : InheritableAttr {
  let Spellings = [GNU<"init_priority">];
  let Args = [UnsignedArgument<"Priority">];
  let Subjects = SubjectList<[Var], ErrorDiag>;
}

def Section : InheritableAttr {
  let Spellings = [GCC<"section">];
  let Args = [StringArgument<"Name">];
  let Subjects = SubjectList<[Function, GlobalVar, 
                              ObjCMethod, ObjCProperty], ErrorDiag,
                             "ExpectedFunctionGlobalVarMethodOrProperty">;
}

def Sentinel : InheritableAttr {
  let Spellings = [GCC<"sentinel">];
  let Args = [DefaultIntArgument<"Sentinel", 0>,
              DefaultIntArgument<"NullPos", 0>];
//  let Subjects = SubjectList<[Function, ObjCMethod, Block, Var]>;
}

def StdCall : InheritableAttr {
  let Spellings = [GCC<"stdcall">, Keyword<"__stdcall">, Keyword<"_stdcall">];
//  let Subjects = [Function, ObjCMethod];
}

def SysVABI : InheritableAttr {
  let Spellings = [GCC<"sysv_abi">];
//  let Subjects = [Function, ObjCMethod];
}

def ThisCall : InheritableAttr {
  let Spellings = [GCC<"thiscall">, Keyword<"__thiscall">,
                   Keyword<"_thiscall">];
//  let Subjects = [Function, ObjCMethod];
}

def Pascal : InheritableAttr {
  let Spellings = [GNU<"pascal">, Keyword<"__pascal">, Keyword<"_pascal">];
//  let Subjects = [Function, ObjCMethod];
}

def TransparentUnion : InheritableAttr {
  let Spellings = [GCC<"transparent_union">];
//  let Subjects = SubjectList<[Record, TypedefName]>;
}

def Unavailable : InheritableAttr {
  let Spellings = [GNU<"unavailable">];
  let Args = [StringArgument<"Message", 1>];
}

def ArcWeakrefUnavailable : InheritableAttr {
  let Spellings = [GNU<"objc_arc_weak_reference_unavailable">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
}

def ObjCGC : TypeAttr {
  let Spellings = [GNU<"objc_gc">];
  let Args = [IdentifierArgument<"Kind">];
}

def ObjCOwnership : InheritableAttr {
  let Spellings = [GNU<"objc_ownership">];
  let Args = [IdentifierArgument<"Kind">];
  let ASTNode = 0;
}

def ObjCRequiresPropertyDefs : InheritableAttr {
  let Spellings = [GNU<"objc_requires_property_definitions">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
}

def Unused : InheritableAttr {
  let Spellings = [GCC<"unused">];
  let Subjects = SubjectList<[Var, ObjCIvar, Type, Label, Field, ObjCMethod,
                              FunctionLike], WarnDiag,
                             "ExpectedVariableFunctionOrLabel">;
}

def Used : InheritableAttr {
  let Spellings = [GCC<"used">];
}

def Uuid : InheritableAttr {
  let Spellings = [Declspec<"uuid">];
  let Args = [StringArgument<"Guid">];
//  let Subjects = SubjectList<[CXXRecord]>;
  let LangOpts = [MicrosoftExt, Borland];
}

def VectorSize : TypeAttr {
  let Spellings = [GCC<"vector_size">];
  let Args = [ExprArgument<"NumBytes">];
}

def VecTypeHint : InheritableAttr {
  let Spellings = [GNU<"vec_type_hint">];
  let Args = [TypeArgument<"TypeHint">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def Visibility : InheritableAttr {
  let Clone = 0;
  let Spellings = [GCC<"visibility">];
  let Args = [EnumArgument<"Visibility", "VisibilityType",
                           ["default", "hidden", "internal", "protected"],
                           ["Default", "Hidden", "Hidden", "Protected"]>];
}

def TypeVisibility : InheritableAttr {
  let Clone = 0;
  let Spellings = [GNU<"type_visibility">, CXX11<"clang", "type_visibility">];
  let Args = [EnumArgument<"Visibility", "VisibilityType",
                           ["default", "hidden", "internal", "protected"],
                           ["Default", "Hidden", "Hidden", "Protected"]>];
//  let Subjects = [Tag, ObjCInterface, Namespace];
}

def VecReturn : InheritableAttr {
  let Spellings = [GNU<"vecreturn">];
  let Subjects = SubjectList<[CXXRecord], ErrorDiag>;
}

def WarnUnused : InheritableAttr {
  let Spellings = [GNU<"warn_unused">];
  let Subjects = SubjectList<[Record]>;
}

def WarnUnusedResult : InheritableAttr {
  let Spellings = [GCC<"warn_unused_result">,
                   CXX11<"clang", "warn_unused_result">];
  let Subjects = SubjectList<[ObjCMethod, CXXRecord, FunctionLike], WarnDiag,
                             "ExpectedFunctionMethodOrClass">;
}

def Weak : InheritableAttr {
  let Spellings = [GCC<"weak">];
  let Subjects = SubjectList<[Var, Function, CXXRecord]>;
}

def WeakImport : InheritableAttr {
  let Spellings = [GNU<"weak_import">];
}

def WeakRef : InheritableAttr {
  let Spellings = [GCC<"weakref">];
  // A WeakRef that has an argument is treated as being an AliasAttr
  let Args = [StringArgument<"Aliasee", 1>];
  let Subjects = SubjectList<[Var, Function], ErrorDiag>;
}

def X86ForceAlignArgPointer : InheritableAttr, TargetSpecificAttr<TargetX86> {
  let Spellings = [GNU<"force_align_arg_pointer">];
  // Technically, this appertains to a FunctionDecl, but the target-specific
  // code silently allows anything function-like (such as typedefs or function
  // pointers), but does not apply the attribute to them.
}

// Attribute to disable AddressSanitizer (or equivalent) checks.
def NoSanitizeAddress : InheritableAttr {
  let Spellings = [GCC<"no_address_safety_analysis">,
                   GCC<"no_sanitize_address">];
  let Subjects = SubjectList<[Function, FunctionTemplate], ErrorDiag>;
}

// Attribute to disable ThreadSanitizer checks.
def NoSanitizeThread : InheritableAttr {
  let Spellings = [GNU<"no_sanitize_thread">];
  let Subjects = SubjectList<[Function, FunctionTemplate], ErrorDiag>;
}

// Attribute to disable MemorySanitizer checks.
def NoSanitizeMemory : InheritableAttr {
  let Spellings = [GNU<"no_sanitize_memory">];
  let Subjects = SubjectList<[Function, FunctionTemplate], ErrorDiag>;
}

// C/C++ Thread safety attributes (e.g. for deadlock, data race checking)

def GuardedVar : InheritableAttr {
  let Spellings = [GNU<"guarded_var">];
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def PtGuardedVar : InheritableAttr {
  let Spellings = [GNU<"pt_guarded_var">];
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def Lockable : InheritableAttr {
  let Spellings = [GNU<"lockable">];
  let Subjects = SubjectList<[Record]>;
}

def ScopedLockable : InheritableAttr {
  let Spellings = [GNU<"scoped_lockable">];
  let Subjects = SubjectList<[Record]>;
}

def NoThreadSafetyAnalysis : InheritableAttr {
  let Spellings = [GNU<"no_thread_safety_analysis">];
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def GuardedBy : InheritableAttr {
  let Spellings = [GNU<"guarded_by">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def PtGuardedBy : InheritableAttr {
  let Spellings = [GNU<"pt_guarded_by">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def AcquiredAfter : InheritableAttr {
  let Spellings = [GNU<"acquired_after">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def AcquiredBefore : InheritableAttr {
  let Spellings = [GNU<"acquired_before">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def ExclusiveLockFunction : InheritableAttr {
  let Spellings = [GNU<"exclusive_lock_function">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def SharedLockFunction : InheritableAttr {
  let Spellings = [GNU<"shared_lock_function">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def AssertExclusiveLock : InheritableAttr {
  let Spellings = [GNU<"assert_exclusive_lock">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def AssertSharedLock : InheritableAttr {
  let Spellings = [GNU<"assert_shared_lock">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

// The first argument is an integer or boolean value specifying the return value
// of a successful lock acquisition.
def ExclusiveTrylockFunction : InheritableAttr {
  let Spellings = [GNU<"exclusive_trylock_function">];
  let Args = [ExprArgument<"SuccessValue">, VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

// The first argument is an integer or boolean value specifying the return value
// of a successful lock acquisition.
def SharedTrylockFunction : InheritableAttr {
  let Spellings = [GNU<"shared_trylock_function">];
  let Args = [ExprArgument<"SuccessValue">, VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def UnlockFunction : InheritableAttr {
  let Spellings = [GNU<"unlock_function">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def LockReturned : InheritableAttr {
  let Spellings = [GNU<"lock_returned">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def LocksExcluded : InheritableAttr {
  let Spellings = [GNU<"locks_excluded">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def ExclusiveLocksRequired : InheritableAttr {
  let Spellings = [GNU<"exclusive_locks_required">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def SharedLocksRequired : InheritableAttr {
  let Spellings = [GNU<"shared_locks_required">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

// C/C++ consumed attributes.

def Consumable : InheritableAttr {
  let Spellings = [GNU<"consumable">];
  let Subjects = SubjectList<[CXXRecord]>;
  let Args = [EnumArgument<"DefaultState", "ConsumedState",
                           ["unknown", "consumed", "unconsumed"],
                           ["Unknown", "Consumed", "Unconsumed"]>];
}

def ConsumableAutoCast : InheritableAttr {
  let Spellings = [GNU<"consumable_auto_cast_state">];
  let Subjects = SubjectList<[CXXRecord]>;
}

def ConsumableSetOnRead : InheritableAttr {
  let Spellings = [GNU<"consumable_set_state_on_read">];
  let Subjects = SubjectList<[CXXRecord]>;
}

def CallableWhen : InheritableAttr {
  let Spellings = [GNU<"callable_when">];
  let Subjects = SubjectList<[CXXMethod]>;
  let Args = [VariadicEnumArgument<"CallableState", "ConsumedState",
                                   ["unknown", "consumed", "unconsumed"],
                                   ["Unknown", "Consumed", "Unconsumed"]>];
}

def ParamTypestate : InheritableAttr {
  let Spellings = [GNU<"param_typestate">];
  let Subjects = SubjectList<[ParmVar]>;
  let Args = [EnumArgument<"ParamState", "ConsumedState",
                           ["unknown", "consumed", "unconsumed"],
                           ["Unknown", "Consumed", "Unconsumed"]>];
}

def ReturnTypestate : InheritableAttr {
  let Spellings = [GNU<"return_typestate">];
  let Subjects = SubjectList<[Function, ParmVar]>;
  let Args = [EnumArgument<"State", "ConsumedState",
                           ["unknown", "consumed", "unconsumed"],
                           ["Unknown", "Consumed", "Unconsumed"]>];
}

def SetTypestate : InheritableAttr {
  let Spellings = [GNU<"set_typestate">];
  let Subjects = SubjectList<[CXXMethod]>;
  let Args = [EnumArgument<"NewState", "ConsumedState",
                           ["unknown", "consumed", "unconsumed"],
                           ["Unknown", "Consumed", "Unconsumed"]>];
}

def TestTypestate : InheritableAttr {
  let Spellings = [GNU<"test_typestate">];
  let Subjects = SubjectList<[CXXMethod]>;
  let Args = [EnumArgument<"TestState", "ConsumedState",
                           ["consumed", "unconsumed"],
                           ["Consumed", "Unconsumed"]>];
}

// Type safety attributes for `void *' pointers and type tags.

def ArgumentWithTypeTag : InheritableAttr {
  let Spellings = [GNU<"argument_with_type_tag">,
                   GNU<"pointer_with_type_tag">];
  let Args = [IdentifierArgument<"ArgumentKind">,
              UnsignedArgument<"ArgumentIdx">,
              UnsignedArgument<"TypeTagIdx">,
              BoolArgument<"IsPointer">];
  let HasCustomParsing = 1;
}

def TypeTagForDatatype : InheritableAttr {
  let Spellings = [GNU<"type_tag_for_datatype">];
  let Args = [IdentifierArgument<"ArgumentKind">,
              TypeArgument<"MatchingCType">,
              BoolArgument<"LayoutCompatible">,
              BoolArgument<"MustBeNull">];
//  let Subjects = SubjectList<[Var], ErrorDiag>;
  let HasCustomParsing = 1;
}

// Microsoft-related attributes

def MsProperty : IgnoredAttr {
  let Spellings = [Declspec<"property">];
}

def MsStruct : InheritableAttr {
  let Spellings = [GCC<"ms_struct">];
  let Subjects = SubjectList<[Record]>;
}

def DLLExport : InheritableAttr, TargetSpecificAttr<TargetX86Win> {
  let Spellings = [Declspec<"dllexport">, GCC<"dllexport">];
  let Subjects = SubjectList<[Function, Var]>;
}

def DLLImport : InheritableAttr, TargetSpecificAttr<TargetX86Win> {
  let Spellings = [Declspec<"dllimport">, GCC<"dllimport">];
  // Technically, the subjects for DllImport are Function and Var, but there is
  // custom semantic handling required when MicrosoftExt is true.
}

def ForceInline : InheritableAttr {
  let Spellings = [Keyword<"__forceinline">];
  let LangOpts = [MicrosoftExt];
}

def SelectAny : InheritableAttr {
  let Spellings = [Declspec<"selectany">];
  let LangOpts = [MicrosoftExt];
}

def Win64 : IgnoredAttr {
  let Spellings = [Keyword<"__w64">];
  let LangOpts = [MicrosoftExt];
}

def Ptr32 : TypeAttr {
  let Spellings = [Keyword<"__ptr32">];
}

def Ptr64 : TypeAttr {
  let Spellings = [Keyword<"__ptr64">];
}

def SPtr : TypeAttr {
  let Spellings = [Keyword<"__sptr">];
}

def UPtr : TypeAttr {
  let Spellings = [Keyword<"__uptr">];
}

def MSInheritance : InheritableAttr {
  let LangOpts = [MicrosoftExt];
  let Spellings = [Keyword<"__single_inheritance">,
                   Keyword<"__multiple_inheritance">,
                   Keyword<"__virtual_inheritance">,
                   Keyword<"__unspecified_inheritance">];
  let AdditionalMembers = [{
  static bool hasVBPtrOffsetField(Spelling Inheritance) {
    return Inheritance == Keyword_unspecified_inheritance;
  }

  // Only member pointers to functions need a this adjustment, since it can be
  // combined with the field offset for data pointers.
  static bool hasNVOffsetField(bool IsMemberFunction, Spelling Inheritance) {
    return IsMemberFunction && Inheritance >= Keyword_multiple_inheritance;
  }

  static bool hasVBTableOffsetField(Spelling Inheritance) {
    return Inheritance >= Keyword_virtual_inheritance;
  }

  static bool hasOnlyOneField(bool IsMemberFunction,
                              Spelling Inheritance) {
    if (IsMemberFunction)
      return Inheritance <= Keyword_single_inheritance;
    return Inheritance <= Keyword_multiple_inheritance;
  }
  }];
}

def Unaligned : IgnoredAttr {
  let Spellings = [Keyword<"__unaligned">];
}
