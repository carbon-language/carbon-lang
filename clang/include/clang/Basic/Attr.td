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
                                S->getKind() != Decl::ImplicitParam &&
                                S->getKind() != Decl::ParmVar &&
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
class TypeArgument<string name> : Argument<name>;
class UnsignedArgument<string name> : Argument<name>;
class VariadicUnsignedArgument<string name> : Argument<name>;

// This one's a doozy, so it gets its own special type
// It can be an unsigned integer, or a type. Either can
// be dependent.
class AlignedArgument<string name> : Argument<name>;

// An integer argument with a default value
class DefaultIntArgument<string name, int default> : IntArgument<name> {
  int Default = default;
}

// This argument is more complex, it includes the enumerator type name,
// a list of strings to accept, and a list of enumerators to map them to.
class EnumArgument<string name, string type, list<string> values,
                         list<string> enums> : Argument<name> {
  string Type = type;
  list<string> Values = values;
  list<string> Enums = enums;
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
  // Any additional text that should be included verbatim in the class.
  code AdditionalMembers = [{}];
}

class InheritableAttr : Attr;

//
// Attributes begin here
//

def Alias : InheritableAttr {
  let Spellings = ["alias"];
  let Args = [StringArgument<"Aliasee">];
}

def Aligned : InheritableAttr {
  let Spellings = ["align", "aligned"];
  let Subjects = [NonBitField, NormalVar, Tag];
  let Args = [AlignedArgument<"Alignment">];
  let Namespaces = ["", "std"];
}

def AlignMac68k : InheritableAttr {
  let Spellings = [];
}

def AlwaysInline : InheritableAttr {
  let Spellings = ["always_inline"];
}

def AnalyzerNoReturn : InheritableAttr {
  let Spellings = ["analyzer_noreturn"];
}

def Annotate : InheritableAttr {
  let Spellings = ["annotate"];
  let Args = [StringArgument<"Annotation">];
}

def AsmLabel : InheritableAttr {
  let Spellings = [];
  let Args = [StringArgument<"Label">];
}

def Blocks : InheritableAttr {
  let Spellings = ["blocks"];
  let Args = [EnumArgument<"Type", "BlockType", ["byref"], ["ByRef"]>];
}

def CarriesDependency : InheritableAttr {
  let Spellings = ["carries_dependency"];
  let Subjects = [ParmVar, Function];
  let Namespaces = ["", "std"];
}

def CDecl : InheritableAttr {
  let Spellings = ["cdecl", "__cdecl"];
}

def CFReturnsRetained : InheritableAttr {
  let Spellings = ["cf_returns_retained"];
  let Subjects = [ObjCMethod, Function];
}

def CFReturnsNotRetained : InheritableAttr {
  let Spellings = ["cf_returns_not_retained"];
  let Subjects = [ObjCMethod, Function];
}

def CFConsumed : InheritableAttr {
  let Spellings = ["cf_consumed"];
  let Subjects = [ParmVar];
}

def Cleanup : InheritableAttr {
  let Spellings = ["cleanup"];
  let Args = [FunctionArgument<"FunctionDecl">];
}

def Common : InheritableAttr {
  let Spellings = ["common"];
}

def Const : InheritableAttr {
  let Spellings = ["const"];
}

def Constructor : InheritableAttr {
  let Spellings = ["constructor"];
  let Args = [IntArgument<"Priority">];
}

def CUDAConstant : InheritableAttr {
  let Spellings = ["constant"];
}

def CUDADevice : Attr {
  let Spellings = ["device"];
}

def CUDAGlobal : InheritableAttr {
  let Spellings = ["global"];
}

def CUDAHost : Attr {
  let Spellings = ["host"];
}

def CUDALaunchBounds : InheritableAttr {
  let Spellings = ["launch_bounds"];
  let Args = [IntArgument<"MaxThreads">, DefaultIntArgument<"MinBlocks", 0>];
}

def CUDAShared : InheritableAttr {
  let Spellings = ["shared"];
}

def Deprecated : InheritableAttr {
  let Spellings = ["deprecated"];
  let Args = [StringArgument<"Message">];
}

def Destructor : InheritableAttr {
  let Spellings = ["destructor"];
  let Args = [IntArgument<"Priority">];
}

def DLLExport : InheritableAttr {
  let Spellings = ["dllexport"];
}

def DLLImport : InheritableAttr {
  let Spellings = ["dllimport"];
}

def Explicit : InheritableAttr {
  let Spellings = [];
}

def FastCall : InheritableAttr {
  let Spellings = ["fastcall", "__fastcall"];
}

def Final : InheritableAttr { 
  let Spellings = [];
}

def Format : InheritableAttr {
  let Spellings = ["format"];
  let Args = [StringArgument<"Type">, IntArgument<"FormatIdx">,
              IntArgument<"FirstArg">];
}

def FormatArg : InheritableAttr {
  let Spellings = ["format_arg"];
  let Args = [IntArgument<"FormatIdx">];
}

def GNUInline : InheritableAttr {
  let Spellings = ["gnu_inline"];
}

def IBAction : InheritableAttr {
  let Spellings = ["ibaction"];
}

def IBOutlet : InheritableAttr {
  let Spellings = ["iboutlet"];
}

def IBOutletCollection : InheritableAttr {
  let Spellings = ["iboutletcollection"];
  let Args = [TypeArgument<"Interface">];
}

def Malloc : InheritableAttr {
  let Spellings = ["malloc"];
}

def MaxFieldAlignment : InheritableAttr {
  let Spellings = [];
  let Args = [UnsignedArgument<"Alignment">];
}

def MayAlias : InheritableAttr {
  let Spellings = ["may_alias"];
}

def MSP430Interrupt : InheritableAttr {
  let Spellings = [];
  let Args = [UnsignedArgument<"Number">];
}

def MBlazeInterruptHandler : InheritableAttr {
  let Spellings = [];
}

def MBlazeSaveVolatiles : InheritableAttr {
  let Spellings = [];
}

def Naked : InheritableAttr {
  let Spellings = ["naked"];
}

def NoCommon : InheritableAttr {
  let Spellings = ["nocommon"];
}

def NoDebug : InheritableAttr {
  let Spellings = ["nodebug"];
}

def NoInline : InheritableAttr {
  let Spellings = ["noinline"];
}

def NonNull : InheritableAttr {
  let Spellings = ["nonnull"];
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
  let Spellings = ["noreturn"];
  // FIXME: Does GCC allow this on the function instead?
  let Subjects = [Function];
  let Namespaces = ["", "std"];
}

def NoInstrumentFunction : InheritableAttr {
  let Spellings = ["no_instrument_function"];
  let Subjects = [Function];
}

def NoThrow : InheritableAttr {
  let Spellings = ["nothrow"];
}

def NSReturnsRetained : InheritableAttr {
  let Spellings = ["ns_returns_retained"];
  let Subjects = [ObjCMethod, Function];
}

def NSReturnsNotRetained : InheritableAttr {
  let Spellings = ["ns_returns_not_retained"];
  let Subjects = [ObjCMethod, Function];
}

def NSReturnsAutoreleased : InheritableAttr {
  let Spellings = ["ns_returns_autoreleased"];
  let Subjects = [ObjCMethod, Function];
}

def NSConsumesSelf : InheritableAttr {
  let Spellings = ["ns_consumes_self"];
  let Subjects = [ObjCMethod];
}

def NSConsumed : InheritableAttr {
  let Spellings = ["ns_consumed"];
  let Subjects = [ParmVar];
}

def ObjCException : InheritableAttr {
  let Spellings = ["objc_exception"];
}

def ObjCNSObject : InheritableAttr {
  let Spellings = ["NSObject"];
}

def Overloadable : Attr {
  let Spellings = ["overloadable"];
}

def Override : InheritableAttr { 
  let Spellings = [];
}

def Ownership : InheritableAttr {
  let Spellings = ["ownership_holds", "ownership_returns", "ownership_takes"];
  let Args = [EnumArgument<"OwnKind", "OwnershipKind",
                    ["ownership_holds", "ownership_returns", "ownership_takes"],
                    ["Holds", "Returns", "Takes"]>,
              StringArgument<"Module">, VariadicUnsignedArgument<"Args">];
}

def Packed : InheritableAttr {
  let Spellings = ["packed"];
}

def Pure : InheritableAttr {
  let Spellings = ["pure"];
}

def Regparm : InheritableAttr {
  let Spellings = ["regparm"];
  let Args = [UnsignedArgument<"NumParams">];
}

def ReqdWorkGroupSize : InheritableAttr {
  let Spellings = ["reqd_work_group_size"];
  let Args = [UnsignedArgument<"XDim">, UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
}

def InitPriority : InheritableAttr {
  let Spellings = ["init_priority"];
  let Args = [UnsignedArgument<"Priority">];
}

def Section : InheritableAttr {
  let Spellings = ["section"];
  let Args = [StringArgument<"Name">];
}

def Sentinel : InheritableAttr {
  let Spellings = ["sentinel"];
  let Args = [DefaultIntArgument<"Sentinel", 0>,
              DefaultIntArgument<"NullPos", 0>];
}

def StdCall : InheritableAttr {
  let Spellings = ["stdcall", "__stdcall"];
}

def ThisCall : InheritableAttr {
  let Spellings = ["thiscall", "__thiscall"];
}

def Pascal : InheritableAttr {
  let Spellings = ["pascal", "__pascal"];
}

def TransparentUnion : InheritableAttr {
  let Spellings = ["transparent_union"];
}

def Unavailable : InheritableAttr {
  let Spellings = ["unavailable"];
  let Args = [StringArgument<"Message">];
}

def Unused : InheritableAttr {
  let Spellings = ["unused"];
}

def Used : InheritableAttr {
  let Spellings = ["used"];
}

def Uuid : InheritableAttr {
  let Spellings = ["uuid"];
  let Args = [StringArgument<"Guid">];
  let Subjects = [CXXRecord];
}

def Visibility : InheritableAttr {
  let Spellings = ["visibility"];
  let Args = [EnumArgument<"Visibility", "VisibilityType",
                           ["default", "hidden", "internal", "protected"],
                           ["Default", "Hidden", "Hidden", "Protected"]>];
}

def VecReturn : InheritableAttr {
  let Spellings = ["vecreturn"];
  let Subjects = [CXXRecord];
}

def WarnUnusedResult : InheritableAttr {
  let Spellings = ["warn_unused_result"];
}

def Weak : InheritableAttr {
  let Spellings = ["weak"];
}

def WeakImport : InheritableAttr {
  let Spellings = ["weak_import"];
}

def WeakRef : InheritableAttr {
  let Spellings = ["weakref"];
}

def X86ForceAlignArgPointer : InheritableAttr {
  let Spellings = [];
}
