import Phases
import Tools

###

class ToolChain(object):
    """ToolChain - Provide mappings of Actions to Tools."""

    def __init__(self, driver):
        self.driver = driver

    def selectTool(self, action):
        """selectTool - Return a Tool instance to use for handling
        some particular action."""
        abstract

class Darwin_ToolChain(ToolChain):
    def __init__(self, driver):
        super(Darwin_ToolChain, self).__init__(driver)
        self.toolMap = {
            Phases.PreprocessPhase : Tools.GCC_PreprocessTool(),
            Phases.CompilePhase : Tools.GCC_CompileTool(),
            Phases.PrecompilePhase : Tools.GCC_PrecompileTool(),
            Phases.AssemblePhase : Tools.DarwinAssembleTool(),
            Phases.LinkPhase : Tools.Collect2Tool(),
            Phases.LipoPhase : Tools.LipoTool(),
            }

    def selectTool(self, action):
        assert isinstance(action, Phases.JobAction)
        return self.toolMap[action.phase.__class__]

class Generic_GCC_ToolChain(ToolChain):
    """Generic_GCC_ToolChain - A tool chain using the 'gcc' command to
    perform all subcommands; this relies on gcc translating the
    options appropriately."""

    def __init__(self, driver):
        super(Generic_GCC_ToolChain, self).__init__(driver)
        self.toolMap = {
            Phases.PreprocessPhase : Tools.GCC_PreprocessTool(),
            Phases.CompilePhase : Tools.GCC_CompileTool(),
            Phases.PrecompilePhase : Tools.GCC_PrecompileTool(),
            Phases.AssemblePhase : Tools.GCC_AssembleTool(),
            Phases.LinkPhase : Tools.GCC_LinkTool(),
            }

    def selectTool(self, action):
        assert isinstance(action, Phases.JobAction)
        return self.toolMap[action.phase.__class__]
