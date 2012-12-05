include "llvm/Option/OptParser.td"

def A : Flag<["-"], "A">, HelpText<"The A option">;
def B : Joined<["-"], "B">, HelpText<"The B option">, MetaVarName<"B">;
def C : Separate<["-"], "C">, HelpText<"The C option">, MetaVarName<"C">;
def D : CommaJoined<["-"], "D">, HelpText<"The D option">, MetaVarName<"D">;
def E : MultiArg<["-"], "E", 2>;
def F : JoinedOrSeparate<["-"], "F">, HelpText<"The F option">, MetaVarName<"F">;
def G : JoinedAndSeparate<["-"], "G">, HelpText<"The G option">, MetaVarName<"G">;

def Ceq : Joined<["-", "--"], "C=">, Alias<C>;

def H : Flag<["-"], "H">, Flags<[HelpHidden]>;
