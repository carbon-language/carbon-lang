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

class Darwin_X86_ToolChain(ToolChain):
    def __init__(self, driver, darwinVersion, gccVersion):
        super(Darwin_X86_ToolChain, self).__init__(driver)
        assert isinstance(darwinVersion, tuple) and len(darwinVersion) == 3
        assert isinstance(gccVersion, tuple) and len(gccVersion) == 3
        self.darwinVersion = darwinVersion
        self.gccVersion = gccVersion

        self.toolMap = {
            Phases.PreprocessPhase : Tools.GCC_PreprocessTool(),
            Phases.CompilePhase : Tools.GCC_CompileTool(),
            Phases.PrecompilePhase : Tools.GCC_PrecompileTool(),
            Phases.AssemblePhase : Tools.Darwin_AssembleTool(self),
            Phases.LinkPhase : Tools.Darwin_X86_LinkTool(self),
            Phases.LipoPhase : Tools.LipoTool(),
            }

    def getToolChainDir(self):
        return 'i686-apple-darwin%d/%s' % (self.darwinVersion[0],
                                           '.'.join(map(str,self.gccVersion)))

    def getProgramPath(self, name):
        # FIXME: Implement proper search.
        return '/usr/libexec/gcc/%s/%s' % (self.getToolChainDir(), name)

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
