////////////////////////////////////////////////////////////////////////////////
// Note: This file is a work in progress. Please do not apply non-trivial
// updates unless you have talked to Sean Hunt <rideau3@gmail.com> prior.
// Merely adding a new attribute is a trivial update.
////////////////////////////////////////////////////////////////////////////////

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
// The description is used in output messages to specify what the subject
// represents. FIXME: Deal with translation issues.
//
// The code fragment is a boolean expression that will confirm that the subject
// meets the requirements; the subject will have the name S, and will have the
// type specified by the base. It should be a simple boolean expression.
class SubsetSubject<AttrSubject base, string description, code check>
    : AttrSubject {
  AttrSubject Base = base;
  string Description = description;
  code CheckCode = check;
}

// This is the type of a variable which C++0x defines [[aligned()]] as being
// a possible subject.
def NormalVar : SubsetSubject<Var, "non-register, non-parameter variable",
                              [{S->getStorageClass() != VarDecl::Register &&
                                S->getKind() != Decl::ImplicitParam
                                S->getKind() != Decl::ParmVar
                                S->getKind() != Decl::NonTypeTemplateParm}]>;
def CXXVirtualMethod : SubsetSubject<CXXRecord, "virtual member function",
                                     [{S->isVirtual()}]>;
def NonBitField : SubsetSubject<Field, "non-bit field",
                                [{!S->isBitField()}]>;

// A single argument to an attribute
class Argument<string name> {
  string Name = name;
}

class IdentifierArgument<string name> : Argument<name>;
class IntArgument<string name> : Argument<name>;
class StringArgument<string name> : Argument<name>;
class ExprArgument<string name> : Argument<name>;
class FunctionArgument<string name> : Argument<name>;
class ObjCInterfaceArgument<string name> : Argument<name>;
class UnsignedIntArgument<string name> : Argument<name>;
class UnsignedIntOrTypeArgument<string name> : Argument<name>;

// An integer argument with a default value
class DefaultIntArgument<string name, int default> : IntArgument<name> {
  int Default = default;
}

// Zero or more arguments of a type
class VariadicArgument<Argument arg> : Argument<arg.Name> {
  Argument VariadicArg = arg;
}

class Attr {
  // The various ways in which an attribute can be spelled in source
  list<string> Spellings;
  // The things to which an attribute can appertain
  list<AttrSubject> Subjects;
  // The arguments allowed on an attribute
  list<Argument> Args = [];
  // The namespaces in which the attribute appears in C++0x attributes.
  // The attribute will not be permitted in C++0x attribute-specifiers if
  // this is empty; the empty string can be used as a namespace.
  list<string> Namespaces = [];
  // A temporary development bit to tell TableGen not to emit certain
  // information about the attribute.
  bit DoNotEmit = 1;
}

//
// Attributes begin here
//

def Alias : Attr {
  let Spellings = ["alias"];
  let Args = [StringArgument<"AliasName">];
}

def Aligned : Attr {
  let Spellings = ["align", "aligned"];
  let Subjects = [NonBitField, NormalVar, Tag];
  let Args = [UnsignedIntOrTypeArgument<"Alignment">];
  let Namespaces = ["", "std"];
}

def AlignMac68k : Attr {
  let Spellings = [];
}

def AlwaysInline : Attr {
  let Spellings = ["always_inline"];
}

def AnalyzerNoReturn : Attr {
  let Spellings = ["analyzer_noreturn"];
}

def Annotate : Attr {
  let Spellings = ["annotate"];
  let Args = [StringArgument<"Annotation">];
}

def AsmLabel : Attr {
  let Spellings = [];
  let Args = [StringArgument<"Label">];
}

def BaseCheck : Attr {
  let Spellings = ["base_check"];
  let Subjects = [CXXRecord];
  let Namespaces = ["", "std"];
  let DoNotEmit = 0;
}

def Blocks : Attr {
  let Spellings = ["blocks"];
  let Args = [IdentifierArgument<"Type">];
}

def CarriesDependency : Attr {
  let Spellings = ["carries_dependency"];
  let Subjects = [ParmVar, Function];
  let Namespaces = ["", "std"];
  let DoNotEmit = 0;
}

def CDecl : Attr {
  let Spellings = ["cdecl", "__cdecl"];
}

def CFReturnsRetained : Attr {
  let Spellings = ["cf_returns_retained"];
}

def CFReturnsNotRetained : Attr {
  let Spellings = ["cf_returns_not_retained"];
}

def Cleanup : Attr {
  let Spellings = ["cleanup"];
  let Args = [FunctionArgument<"FunctionDecl">];
}

def Const : Attr {
  let Spellings = ["const"];
}

def Constructor : Attr {
  let Spellings = ["constructor"];
  let Args = [IntArgument<"Priority">];
}

def Deprecated : Attr {
  let Spellings = ["deprecated"];
}

def Destructor : Attr {
  let Spellings = ["destructor"];
  let Args = [IntArgument<"Priority">];
}

def DLLExport : Attr {
  let Spellings = ["dllexport"];
}

def DLLImport : Attr {
  let Spellings = ["dllimport"];
}

def FastCall : Attr {
  let Spellings = ["fastcall", "__fastcall"];
}

def Final : Attr {
  let Spellings = ["final"];
  let Subjects = [CXXRecord, CXXVirtualMethod];
  let Namespaces = ["", "std"];
  let DoNotEmit = 0;
}

def Format : Attr {
  let Spellings = ["format"];
  let Args = [StringArgument<"Type">, IntArgument<"FormatIdx">,
              IntArgument<"FirstArg">];
}

def FormatArg : Attr {
  let Spellings = ["format_arg"];
  let Args = [IntArgument<"FormatIdx">];
}

def GNUInline : Attr {
  let Spellings = ["gnu_inline"];
}

def Hiding : Attr {
  let Spellings = ["hiding"];
  let Subjects = [Field, CXXMethod];
  let Namespaces = ["", "std"];
  let DoNotEmit = 0;
}

def IBAction : Attr {
  let Spellings = ["ibaction"];
}

def IBOutlet : Attr {
  let Spellings = ["iboutlet"];
}

def IBOutletCollection : Attr {
  let Spellings = ["iboutletcollection"];
  let Args = [ObjCInterfaceArgument<"Class">];
}

def Malloc : Attr {
  let Spellings = ["malloc"];
}

def MaxFieldAlignment : Attr {
  let Spellings = [];
  let Args = [UnsignedIntArgument<"Alignment">];
}

def MSP430Interrupt : Attr {
  let Spellings = [];
  let Args = [UnsignedIntArgument<"Number">];
}

def NoDebug : Attr {
  let Spellings = ["nodebug"];
}

def NoInline : Attr {
  let Spellings = ["noinline"];
}

def NonNull : Attr {
  let Spellings = ["nonnull"];
  let Args = [VariadicArgument<UnsignedIntArgument<"Args">>];
}

def NoReturn : Attr {
  let Spellings = ["noreturn"];
  // FIXME: Does GCC allow this on the function instead?
  let Subjects = [Function];
  let Namespaces = ["", "std"];
}

def NoInstrumentFunction : Attr {
  let Spellings = ["no_instrument_function"];
  let Subjects = [Function];
}

def NoThrow : Attr {
  let Spellings = ["nothrow"];
}

def NSReturnsRetained : Attr {
  let Spellings = ["ns_returns_retained"];
}

def NSReturnsNotRetained : Attr {
  let Spellings = ["ns_returns_not_retained"];
}

def ObjCException : Attr {
  let Spellings = ["objc_exception"];
}

def ObjCNSObject : Attr {
  let Spellings = ["NSOjbect"];
}

def Override : Attr {
  let Spellings = ["override"];
  let Subjects = [CXXVirtualMethod];
  let Namespaces = ["", "std"];
  let DoNotEmit = 0;
}

def Overloadable : Attr {
  let Spellings = ["overloadable"];
}

def OwnershipReturns : Attr {
  let Spellings = ["ownership_returns"];
  let Args = [StringArgument<"Module">, IntArgument<"SizeIdx">];
}

def OwnershipTakes : Attr {
  let Spellings = ["ownership_takes"];
  let Args = [StringArgument<"Module">, IntArgument<"PtrIdx">];
}

def OwnershipHolds : Attr {
  let Spellings = ["ownership_holds"];
  let Args = [StringArgument<"Module">, IntArgument<"PtrIdx">];
}

def Packed : Attr {
  let Spellings = ["packed"];
}

def Pure : Attr {
  let Spellings = ["pure"];
}

def Regparm : Attr {
  let Spellings = ["regparm"];
  let Args = [UnsignedIntArgument<"NumParams">];
}

def ReqdWorkGroupSize : Attr {
  let Spellings = ["reqd_work_group_size"];
  let Args = [UnsignedIntArgument<"XDim">, UnsignedIntArgument<"YDim">,
              UnsignedIntArgument<"ZDim">];
}

def InitPriority : Attr {
  let Spellings = ["init_priority"];
  let Args = [UnsignedIntArgument<"Priority">];
}

def Section : Attr {
  let Spellings = ["section"];
  let Args = [StringArgument<"Name">];
}

def Sentinel : Attr {
  let Spellings = ["sentinel"];
  let Args = [DefaultIntArgument<"NulPos", 0>,
              DefaultIntArgument<"Sentinel", 0>];
}

def StdCall : Attr {
  let Spellings = ["stdcall", "__stdcall"];
}

def ThisCall : Attr {
  let Spellings = ["thiscall", "__thiscall"];
}

def TransparentUnion : Attr {
  let Spellings = ["transparent_union"];
}

def Unavailable : Attr {
  let Spellings = ["unavailable"];
}

def Unused : Attr {
  let Spellings = ["unused"];
}

def Used : Attr {
  let Spellings = ["used"];
}

def Visibility : Attr {
  let Spellings = ["visibility"];
  let Args = [StringArgument<"Visibility">];
}

def VecReturn : Attr {
  let Spellings = ["vecreturn"];
  let Subjects = [CXXRecord];
  let DoNotEmit = 0;
}

def WarnUnusedResult : Attr {
  let Spellings = ["warn_unused_result"];
}

def Weak : Attr {
  let Spellings = ["weak"];
}

def WeakImport : Attr {
  let Spellings = ["weak_import"];
}

def WeakRef : Attr {
  let Spellings = ["weakref"];
}

def X86ForceAlignArgPointer : Attr {
  let Spellings = [];
}
