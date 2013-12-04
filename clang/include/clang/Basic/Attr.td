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

def Struct : SubsetSubject<Record,
                           [{!S->isUnion()}]>;

def TLSVar : SubsetSubject<Var,
                           [{S->getTLSKind() != 0}]>;

def SharedVar : SubsetSubject<Var,
                              [{S->hasGlobalStorage() && !S->getTLSKind()}]>;

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
class SourceLocArgument<string name, bit opt = 0> : Argument<name, opt>;
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
}

class GNU<string name> : Spelling<name, "GNU">;
class Declspec<string name> : Spelling<name, "Declspec">;
class CXX11<string namespace, string name> : Spelling<name, "CXX11"> {
  string Namespace = namespace;
}
class Keyword<string name> : Spelling<name, "Keyword">;

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
  // Lists language options, one of which is required to be true for the
  // attribute to be applicable. If empty, no language options are required.
  list<LangOpt> LangOpts = [];
  // Any additional text that should be included verbatim in the class.
  code AdditionalMembers = [{}];
}

/// A type attribute is not processed on a declaration or a statement.
class TypeAttr : Attr {
  let ASTNode = 0;
}

/// An inheritable attribute is inherited by later redeclarations.
class InheritableAttr : Attr;

/// A target-specific attribute that is meant to be processed via
/// TargetAttributesSema::ProcessDeclAttribute.  This class is meant to be used
/// as a mixin with InheritableAttr or Attr depending on the attribute's needs.
class TargetSpecificAttr;

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
  let Spellings = [GNU<"alias">, CXX11<"gnu", "alias">];
  let Args = [StringArgument<"Aliasee">];
}

def Aligned : InheritableAttr {
  let Spellings = [GNU<"aligned">, Declspec<"align">, CXX11<"gnu", "aligned">,
                   Keyword<"alignas">, Keyword<"_Alignas">];
//  let Subjects = SubjectList<[NonBitField, NormalVar, Tag]>;
  let Args = [AlignedArgument<"Alignment", 1>];
  let Accessors = [Accessor<"isGNU", [GNU<"aligned">, CXX11<"gnu","aligned">]>,
                   Accessor<"isC11", [Keyword<"_Alignas">]>,
                   Accessor<"isAlignas", [Keyword<"alignas">,
                                          Keyword<"_Alignas">]>,
                   Accessor<"isDeclspec",[Declspec<"align">]>];
}

def AlignMac68k : InheritableAttr {
  let Spellings = [];
  let SemaHandler = 0;
}

def AllocSize : InheritableAttr {
  let Spellings = [GNU<"alloc_size">, CXX11<"gnu", "alloc_size">];
  let Args = [VariadicUnsignedArgument<"Args">];
}

def AlwaysInline : InheritableAttr {
  let Spellings = [GNU<"always_inline">, CXX11<"gnu", "always_inline">];
  let Subjects = SubjectList<[Function]>;
}

def TLSModel : InheritableAttr {
  let Spellings = [GNU<"tls_model">, CXX11<"gnu", "tls_model">];
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

def ARMInterrupt : InheritableAttr, TargetSpecificAttr {
  let Spellings = [GNU<"interrupt">];
  let Args = [EnumArgument<"Interrupt", "InterruptType",
                           ["IRQ", "FIQ", "SWI", "ABORT", "UNDEF", ""],
                           ["IRQ", "FIQ", "SWI", "ABORT", "UNDEF", "Generic"],
                           1>];
}

def AsmLabel : InheritableAttr {
  let Spellings = [];
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
  let Spellings = [GNU<"cdecl">, CXX11<"gnu", "cdecl">, Keyword<"__cdecl">,
                   Keyword<"_cdecl">];
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
  let Spellings = [GNU<"cleanup">, CXX11<"gnu", "cleanup">];
  let Args = [FunctionArgument<"FunctionDecl">];
  let Subjects = SubjectList<[Var]>;
}

def Cold : InheritableAttr {
  let Spellings = [GNU<"cold">, CXX11<"gnu", "cold">];
  let Subjects = SubjectList<[Function]>;
}

def Common : InheritableAttr {
  let Spellings = [GNU<"common">, CXX11<"gnu", "common">];
  let Subjects = SubjectList<[Var]>;
}

def Const : InheritableAttr {
  let Spellings = [GNU<"const">, GNU<"__const">,
                   CXX11<"gnu", "const">, CXX11<"gnu", "__const">];
}

def Constructor : InheritableAttr {
  let Spellings = [GNU<"constructor">, CXX11<"gnu", "constructor">];
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
}

def OpenCLImageAccess : Attr {
  let Spellings = [GNU<"opencl_image_access">];
  let Args = [IntArgument<"Access">];
}

def Deprecated : InheritableAttr {
  let Spellings = [GNU<"deprecated">,
                   CXX11<"gnu", "deprecated">, CXX11<"","deprecated">];
  let Args = [StringArgument<"Message", 1>];
}

def Destructor : InheritableAttr {
  let Spellings = [GNU<"destructor">, CXX11<"gnu", "destructor">];
  let Args = [DefaultIntArgument<"Priority", 65535>];
  let Subjects = SubjectList<[Function]>;
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
  let Spellings = [GNU<"fastcall">, CXX11<"gnu", "fastcall">,
                   Keyword<"__fastcall">, Keyword<"_fastcall">];
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
  let Spellings = [GNU<"format">, CXX11<"gnu", "format">];
  let Args = [IdentifierArgument<"Type">, IntArgument<"FormatIdx">,
              IntArgument<"FirstArg">];
}

def FormatArg : InheritableAttr {
  let Spellings = [GNU<"format_arg">, CXX11<"gnu", "format_arg">];
  let Args = [IntArgument<"FormatIdx">];
}

def GNUInline : InheritableAttr {
  let Spellings = [GNU<"gnu_inline">, CXX11<"gnu", "gnu_inline">];
  let Subjects = SubjectList<[Function]>;
}

def Hot : InheritableAttr {
  let Spellings = [GNU<"hot">, CXX11<"gnu", "hot">];
  let Subjects = SubjectList<[Function]>;
}

def IBAction : InheritableAttr {
  let Spellings = [GNU<"ibaction">];
  let Subjects = SubjectList<[ObjCInstanceMethod], WarnDiag,
                             "ExpectedObjCInstanceMethod">;
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
  let Spellings = [GNU<"malloc">, CXX11<"gnu", "malloc">];
//  let Subjects = [Function];
}

def MaxFieldAlignment : InheritableAttr {
  let Spellings = [];
  let Args = [UnsignedArgument<"Alignment">];
  let SemaHandler = 0;
}

def MayAlias : InheritableAttr {
  let Spellings = [GNU<"may_alias">, CXX11<"gnu", "may_alias">];
}

def MSABI : InheritableAttr {
  let Spellings = [GNU<"ms_abi">, CXX11<"gnu", "ms_abi">];
//  let Subjects = [Function, ObjCMethod];
}

def MSP430Interrupt : InheritableAttr, TargetSpecificAttr {
  let Spellings = [];
  let Args = [UnsignedArgument<"Number">];
  let SemaHandler = 0;
}

def Mips16 : InheritableAttr, TargetSpecificAttr {
  let Spellings = [GNU<"mips16">, CXX11<"gnu", "mips16">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def Mode : Attr {
  let Spellings = [GNU<"mode">, CXX11<"gnu", "mode">];
  let Args = [IdentifierArgument<"Mode">];
}

def Naked : InheritableAttr {
  let Spellings = [GNU<"naked">, CXX11<"gnu", "naked">];
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
  let Spellings = [GNU<"returns_twice">, CXX11<"gnu", "returns_twice">];
  let Subjects = SubjectList<[Function]>;
}

def NoCommon : InheritableAttr {
  let Spellings = [GNU<"nocommon">, CXX11<"gnu", "nocommon">];
  let Subjects = SubjectList<[Var]>;
}

def NoDebug : InheritableAttr {
  let Spellings = [GNU<"nodebug">];
}

def NoInline : InheritableAttr {
  let Spellings = [GNU<"noinline">, CXX11<"gnu", "noinline">];
  let Subjects = SubjectList<[Function]>;
}

def NoMips16 : InheritableAttr, TargetSpecificAttr {
  let Spellings = [GNU<"nomips16">, CXX11<"gnu", "nomips16">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def NonNull : InheritableAttr {
  let Spellings = [GNU<"nonnull">, CXX11<"gnu", "nonnull">];
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

def NoReturn : InheritableAttr {
  let Spellings = [GNU<"noreturn">, CXX11<"gnu", "noreturn">];
  // FIXME: Does GCC allow this on the function instead?
}

def NoInstrumentFunction : InheritableAttr {
  let Spellings = [GNU<"no_instrument_function">,
                   CXX11<"gnu", "no_instrument_function">];
  let Subjects = SubjectList<[Function]>;
}

def NoThrow : InheritableAttr {
  let Spellings = [GNU<"nothrow">, CXX11<"gnu", "nothrow">];
}

def NSBridged : InheritableAttr {
  let Spellings = [GNU<"ns_bridged">];
  let Subjects = SubjectList<[Struct], ErrorDiag, "ExpectedStruct">;
  let Args = [IdentifierArgument<"BridgedType", 1>];
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

def ObjCSuppressProtocol : InheritableAttr {
  let Spellings = [GNU<"objc_suppress_protocol_methods">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
  let Args = [IdentifierArgument<"Protocol">];
}

def ObjCDesignatedInitializer : Attr {
  let Spellings = [GNU<"objc_designated_initializer">];
  let Subjects = SubjectList<[ObjCMethod], ErrorDiag>;
}

def Overloadable : Attr {
  let Spellings = [GNU<"overloadable">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
}

def Override : InheritableAttr { 
  let Spellings = [];
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
}

def Packed : InheritableAttr {
  let Spellings = [GNU<"packed">, CXX11<"gnu", "packed">];
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
  let Spellings = [GNU<"pcs">, CXX11<"gnu", "pcs">];
  let Args = [EnumArgument<"PCS", "PCSType",
                           ["aapcs", "aapcs-vfp"],
                           ["AAPCS", "AAPCS_VFP"]>];
//  let Subjects = [Function, ObjCMethod];
}

def Pure : InheritableAttr {
  let Spellings = [GNU<"pure">, CXX11<"gnu", "pure">];
}

def Regparm : InheritableAttr {
  let Spellings = [GNU<"regparm">, CXX11<"gnu", "regparm">];
  let Args = [UnsignedArgument<"NumParams">];
}

def ReqdWorkGroupSize : InheritableAttr {
  let Spellings = [GNU<"reqd_work_group_size">];
  let Args = [UnsignedArgument<"XDim">, UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
}

def WorkGroupSizeHint :  InheritableAttr {
  let Spellings = [GNU<"work_group_size_hint">];
  let Args = [UnsignedArgument<"XDim">, 
              UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
}

def InitPriority : InheritableAttr {
  let Spellings = [GNU<"init_priority">];
  let Args = [UnsignedArgument<"Priority">];
  let Subjects = SubjectList<[Var], ErrorDiag>;
}

def Section : InheritableAttr {
  let Spellings = [GNU<"section">, CXX11<"gnu", "section">];
  let Args = [StringArgument<"Name">];
}

def Sentinel : InheritableAttr {
  let Spellings = [GNU<"sentinel">, CXX11<"gnu", "sentinel">];
  let Args = [DefaultIntArgument<"Sentinel", 0>,
              DefaultIntArgument<"NullPos", 0>];
//  let Subjects = SubjectList<[Function, ObjCMethod, Block, Var]>;
}

def StdCall : InheritableAttr {
  let Spellings = [GNU<"stdcall">, CXX11<"gnu", "stdcall">,
                   Keyword<"__stdcall">, Keyword<"_stdcall">];
//  let Subjects = [Function, ObjCMethod];
}

def SysVABI : InheritableAttr {
  let Spellings = [GNU<"sysv_abi">, CXX11<"gnu", "sysv_abi">];
//  let Subjects = [Function, ObjCMethod];
}

def ThisCall : InheritableAttr {
  let Spellings = [GNU<"thiscall">, CXX11<"gnu", "thiscall">,
                   Keyword<"__thiscall">, Keyword<"_thiscall">];
//  let Subjects = [Function, ObjCMethod];
}

def Pascal : InheritableAttr {
  let Spellings = [GNU<"pascal">, Keyword<"__pascal">, Keyword<"_pascal">];
//  let Subjects = [Function, ObjCMethod];
}

def TransparentUnion : InheritableAttr {
  let Spellings = [GNU<"transparent_union">, CXX11<"gnu", "transparent_union">];
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
  let Spellings = [GNU<"unused">, CXX11<"gnu", "unused">];
}

def Used : InheritableAttr {
  let Spellings = [GNU<"used">, CXX11<"gnu", "used">];
}

def Uuid : InheritableAttr {
  let Spellings = [GNU<"uuid">];
  let Args = [StringArgument<"Guid">];
//  let Subjects = SubjectList<[CXXRecord]>;
  let LangOpts = [MicrosoftExt, Borland];
}

def VectorSize : TypeAttr {
  let Spellings = [GNU<"vector_size">, CXX11<"gnu", "vector_size">];
  let Args = [ExprArgument<"NumBytes">];
}

def VecTypeHint : InheritableAttr {
  let Spellings = [GNU<"vec_type_hint">];
  let Args = [TypeArgument<"TypeHint">];
}

def Visibility : InheritableAttr {
  let Clone = 0;
  let Spellings = [GNU<"visibility">, CXX11<"gnu", "visibility">];
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
  let Spellings = [GNU<"warn_unused_result">,
                   CXX11<"clang", "warn_unused_result">,
                   CXX11<"gnu", "warn_unused_result">];
}

def Weak : InheritableAttr {
  let Spellings = [GNU<"weak">, CXX11<"gnu", "weak">];
  let Subjects = SubjectList<[Var, Function, CXXRecord]>;
}

def WeakImport : InheritableAttr {
  let Spellings = [GNU<"weak_import">];
}

def WeakRef : InheritableAttr {
  let Spellings = [GNU<"weakref">, CXX11<"gnu", "weakref">];
  // A WeakRef that has an argument is treated as being an AliasAttr
  let Args = [StringArgument<"Aliasee", 1>];
  let Subjects = SubjectList<[Var, Function], ErrorDiag>;
}

def X86ForceAlignArgPointer : InheritableAttr, TargetSpecificAttr {
  let Spellings = [GNU<"force_align_arg_pointer">];
  // Technically, this appertains to a FunctionDecl, but the target-specific
  // code silently allows anything function-like (such as typedefs or function
  // pointers), but does not apply the attribute to them.
}

// Attribute to disable AddressSanitizer (or equivalent) checks.
def NoSanitizeAddress : InheritableAttr {
  let Spellings = [GNU<"no_address_safety_analysis">,
                   GNU<"no_sanitize_address">,
                   CXX11<"gnu", "no_address_safety_analysis">,
                   CXX11<"gnu", "no_sanitize_address">];
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
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def PtGuardedBy : InheritableAttr {
  let Spellings = [GNU<"pt_guarded_by">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def AcquiredAfter : InheritableAttr {
  let Spellings = [GNU<"acquired_after">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def AcquiredBefore : InheritableAttr {
  let Spellings = [GNU<"acquired_before">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
}

def ExclusiveLockFunction : InheritableAttr {
  let Spellings = [GNU<"exclusive_lock_function">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def SharedLockFunction : InheritableAttr {
  let Spellings = [GNU<"shared_lock_function">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def AssertExclusiveLock : InheritableAttr {
  let Spellings = [GNU<"assert_exclusive_lock">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def AssertSharedLock : InheritableAttr {
  let Spellings = [GNU<"assert_shared_lock">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

// The first argument is an integer or boolean value specifying the return value
// of a successful lock acquisition.
def ExclusiveTrylockFunction : InheritableAttr {
  let Spellings = [GNU<"exclusive_trylock_function">];
  let Args = [ExprArgument<"SuccessValue">, VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

// The first argument is an integer or boolean value specifying the return value
// of a successful lock acquisition.
def SharedTrylockFunction : InheritableAttr {
  let Spellings = [GNU<"shared_trylock_function">];
  let Args = [ExprArgument<"SuccessValue">, VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def UnlockFunction : InheritableAttr {
  let Spellings = [GNU<"unlock_function">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def LockReturned : InheritableAttr {
  let Spellings = [GNU<"lock_returned">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def LocksExcluded : InheritableAttr {
  let Spellings = [GNU<"locks_excluded">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def ExclusiveLocksRequired : InheritableAttr {
  let Spellings = [GNU<"exclusive_locks_required">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let Subjects = SubjectList<[Function, FunctionTemplate]>;
}

def SharedLocksRequired : InheritableAttr {
  let Spellings = [GNU<"shared_locks_required">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
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
  let Spellings = [GNU<"ms_struct">, CXX11<"gnu", "ms_struct">];
  let Subjects = SubjectList<[Record]>;
}

def DLLExport : InheritableAttr, TargetSpecificAttr {
  let Spellings = [Declspec<"dllexport">];
}

def DLLImport : InheritableAttr, TargetSpecificAttr {
  let Spellings = [Declspec<"dllimport">];
}

def ForceInline : InheritableAttr {
  let Spellings = [Keyword<"__forceinline">];
  let LangOpts = [MicrosoftExt];
}

def SelectAny : InheritableAttr {
  let Spellings = [Declspec<"selectany">];
  let LangOpts = [MicrosoftExt];
}

def Win64 : InheritableAttr {
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

class MSInheritanceAttr : InheritableAttr {
  let LangOpts = [MicrosoftExt];
}

def SingleInheritance : MSInheritanceAttr {
  let Spellings = [Keyword<"__single_inheritance">];
}

def MultipleInheritance : MSInheritanceAttr {
  let Spellings = [Keyword<"__multiple_inheritance">];
}

def VirtualInheritance : MSInheritanceAttr {
  let Spellings = [Keyword<"__virtual_inheritance">];
}

// This attribute doesn't have any spellings, but we can apply it implicitly to
// incomplete types that lack any of the other attributes.
def UnspecifiedInheritance : MSInheritanceAttr {
  let Spellings = [];
}

def Unaligned : IgnoredAttr {
  let Spellings = [Keyword<"__unaligned">];
}
