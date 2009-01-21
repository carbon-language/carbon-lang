import Util

class Action(object):
    def __init__(self, inputs, type):
        self.inputs = inputs
        self.type = type

class BindArchAction(Action):
    """BindArchAction - Represent an architecture binding for child
    actions."""

    def __init__(self, input, arch):
        super(BindArchAction, self).__init__([input], input.type)
        self.arch = arch

    def __repr__(self):
        return Util.prefixAndPPrint(self.__class__.__name__,
                                    (self.inputs[0], self.arch))

class InputAction(Action):
    """InputAction - Adapt an input file to an action & type. """

    def __init__(self, filename, type):
        super(InputAction, self).__init__([], type)
        self.filename = filename

    def __repr__(self):
        return Util.prefixAndPPrint(self.__class__.__name__,
                                    (self.filename, self.type))

class JobAction(Action):
    """JobAction - Represent a job tied to a particular compilation
    phase."""

    def __init__(self, phase, inputs, type):
        super(JobAction, self).__init__(inputs, type)
        self.phase = phase

    def __repr__(self):
        return Util.prefixAndPPrint(self.__class__.__name__,
                                    (self.phase, self.inputs, self.type))

###

class Phase(object):
    """Phase - Represent an abstract task in the compilation
    pipeline."""

    eOrderNone = 0
    eOrderPreprocess = 1
    eOrderCompile = 2
    eOrderAssemble = 3
    eOrderPostAssemble = 4
    
    def __init__(self, name, order):
        self.name = name
        self.order = order

    def __repr__(self):
        return Util.prefixAndPPrint(self.__class__.__name__,
                                    (self.name, self.order))

class PreprocessPhase(Phase):
    def __init__(self):
        super(PreprocessPhase, self).__init__("preprocessor", Phase.eOrderPreprocess)

class PrecompilePhase(Phase):
    def __init__(self):
        super(PrecompilePhase, self).__init__("precompiler", Phase.eOrderCompile)

class AnalyzePhase(Phase):
    def __init__(self):
        super(AnalyzePhase, self).__init__("analyze", Phase.eOrderCompile)

class SyntaxOnlyPhase(Phase):
    def __init__(self):
        super(SyntaxOnlyPhase, self).__init__("syntax-only", Phase.eOrderCompile)

class CompilePhase(Phase):
    def __init__(self):
        super(CompilePhase, self).__init__("compiler", Phase.eOrderCompile)

class AssemblePhase(Phase):
    def __init__(self):
        super(AssemblePhase, self).__init__("assembler", Phase.eOrderAssemble)

class LinkPhase(Phase):
    def __init__(self):
        super(LinkPhase, self).__init__("linker", Phase.eOrderPostAssemble)

class LipoPhase(Phase):
    def __init__(self):
        super(LipoPhase, self).__init__("lipo", Phase.eOrderPostAssemble)

