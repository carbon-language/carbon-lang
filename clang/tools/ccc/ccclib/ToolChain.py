import Arguments
import Phases
import Tools
import Types

###

class ToolChain(object):
    """ToolChain - Provide mappings of Actions to Tools."""

    def __init__(self, driver):
        self.driver = driver

    def selectTool(self, action):
        """selectTool - Return a Tool instance to use for handling
        some particular action."""
        abstract

    def translateArgs(self, args, arch):
        """translateArgs - Callback to allow argument translation for
        an entire toolchain."""

        # FIXME: Would be nice to move arch handling out of generic
        # code.
        if arch:
            archName = args.getValue(arch)
            al = Arguments.DerivedArgList(args)
            for arg in args.args:
                if arg.opt is args.parser.archOption:
                    if arg is arch:
                        al.append(arg)
                elif arg.opt is args.parser.XarchOption:
                    if args.getJoinedValue(arg) == archName:
                        # FIXME: Fix this.
                        arg = args.parser.lookupOptForArg(Arguments.InputIndex(0, arg.index.pos + 1),
                                                          args.getSeparateValue(arg),
                                                          iter([]))
                        al.append(arg)
                else:
                    al.append(arg)
            return al
        else:
            return args

class Darwin_X86_ToolChain(ToolChain):
    def __init__(self, driver, darwinVersion, gccVersion, archName):
        super(Darwin_X86_ToolChain, self).__init__(driver)
        assert isinstance(darwinVersion, tuple) and len(darwinVersion) == 3
        assert isinstance(gccVersion, tuple) and len(gccVersion) == 3
        self.darwinVersion = darwinVersion
        self.gccVersion = gccVersion
        self.archName = archName

        self.toolMap = {
            Phases.PreprocessPhase : Tools.GCC_PreprocessTool(),
            Phases.CompilePhase : Tools.Darwin_X86_CompileTool(self),
            Phases.PrecompilePhase : Tools.GCC_PrecompileTool(),
            Phases.AssemblePhase : Tools.Darwin_AssembleTool(self),
            Phases.LinkPhase : Tools.Darwin_X86_LinkTool(self),
            Phases.LipoPhase : Tools.LipoTool(),
            }
        self.clangTool = Tools.Clang_CompileTool()

    def getToolChainDir(self):
        return 'i686-apple-darwin%d/%s' % (self.darwinVersion[0],
                                           '.'.join(map(str,self.gccVersion)))

    def getProgramPath(self, name):
        # FIXME: Implement proper search.
        return '/usr/libexec/gcc/%s/%s' % (self.getToolChainDir(), name)

    def getMacosxVersionMin(self):
        major,minor,minorminor = self.darwinVersion
        return '%d.%d.%d' % (10, major-4, minor)

    def selectTool(self, action):
        assert isinstance(action, Phases.JobAction)
        
        if (self.driver.cccClang and
            self.archName == 'i386' and
            action.inputs[0].type in (Types.CType, Types.CTypeNoPP) and
            isinstance(action.phase, Phases.CompilePhase)):
            return self.clangTool

        return self.toolMap[action.phase.__class__]

    def translateArgs(self, args, arch):
        args = super(Darwin_X86_ToolChain, self).translateArgs(args, arch)
        
        # If arch hasn't been bound we don't need to do anything yet.
        if not arch:
            return args

        al = Arguments.DerivedArgList(args)
        if not args.getLastArg(args.parser.m_macosxVersionMinOption):
            al.append(al.makeJoinedArg(self.getMacosxVersionMin(),
                                       args.parser.m_macosxVersionMinOption))
        for arg in args:
            if arg.opt is args.parser.f_constantCfstringsOption:
                al.append(al.makeFlagArg(args.parser.m_constantCfstringsOption))
            elif arg.opt is args.parser.f_noConstantCfstringsOption:
                al.append(al.makeFlagArg(args.parser.m_noConstantCfstringsOption))
            elif arg.opt is args.parser.WnonportableCfstringsOption:
                al.append(al.makeFlagArg(args.parser.m_warnNonportableCfstringsOption))
            elif arg.opt is args.parser.WnoNonportableCfstringsOption:
                al.append(al.makeFlagArg(args.parser.m_noWarnNonportableCfstringsOption))
            elif arg.opt is args.parser.f_pascalStringsOption:
                al.append(al.makeFlagArg(args.parser.m_pascalStringsOption))
            elif arg.opt is args.parser.f_noPascalStringsOption:
                al.append(al.makeFlagArg(args.parser.m_noPascalStringsOption))
            else:
                al.append(arg)

        # FIXME: Actually, gcc always adds this, but it is filtered
        # for duplicates somewhere. This also changes the order of
        # things, so look it up.
        if arch and args.getValue(arch) == 'x86_64':
            if not args.getLastArg(args.parser.m_64Option):
                al.append(al.makeFlagArg(args.parser.m_64Option))

        if not args.getLastArg(args.parser.m_tuneOption):
            al.append(al.makeJoinedArg('core2',
                                       args.parser.m_tuneOption))

        return al

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
